#!/usr/bin/env python

import os
import torch
import logging
import argparse
import numpy as np
import xarray as xr
from utils import Config
from dataset import DataModule
from models import ModelFactory

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

config = Config()
SPLITDIR   = config.splitdir
MODELDIR   = config.modeldir
PREDSDIR   = config.predsdir
FEATSDIR   = config.featsdir
WEIGHTSDIR = config.weightsdir
FIELDVARS  = config.fieldvars
LOCALVARS  = config.localvars
TARGETVAR  = config.targetvar
LATRANGE   = config.latrange
LONRANGE   = config.lonrange
SEED       = config.seed
WORKERS    = config.workers
BATCHSIZE  = config.batchsize
    
def load(name,modelconfig,result,device,fieldvars=FIELDVARS,localvars=LOCALVARS,modeldir=MODELDIR):
    '''
    Purpose: Initialize a model instance from ModelFactory.build() and populate with weights from a saved checkpoint.
    Args:
    - name (str): model name
    - modelconfig (dict): model configuration
    - result (dict[str,object]): dictionary from DataModule.dataloaders()
    - device (str): device to use
    - fieldvars (list[str]): predictor field variable names (defaults to FIELDVARS)
    - localvars (list[str]): local input variable names (defaults to LOCALVARS)
    - modeldir (str): directory containing checkpoints (defaults to MODELDIR)
    Returns:
    - torch.nn.Module: model with loaded state_dict on 'device' or None if checkpoint not found
    '''
    kind     = modelconfig['kind']
    filedir  = os.path.join(modeldir,kind)
    filename = f'{name}.pth'
    filepath = os.path.join(filedir,filename)
    if not os.path.exists(filepath):
        logger.error(f'   Checkpoint not found: {filepath}')
        return None
    patchshape = result['geometry'].shape()
    nfieldvars = len(fieldvars)
    nlocalvars = len(localvars)
    model = ModelFactory.build(name,modelconfig,patchshape,nfieldvars,nlocalvars)
    state = torch.load(filepath,map_location='cpu')
    model.load_state_dict(state)
    return model.to(device)

def inference(model,split,result,uselocal,device):
    '''
    Purpose: Run inference on the model.
    Args:
    - model (torch.nn.Module): trained model instance 
    - split (str): 'valid' | 'test'
    - result (dict[str,object]): dictionary from DataModule.dataloaders()
    - uselocal (bool): whether to use local inputs
    - device (str): device to use
    Returns:
    - xr.DataArray: predicted precipitation DataArray
    '''
    dataloader = result['loaders'][split]
    model.eval()
    outputslist  = []
    featureslist = []
    with torch.no_grad():
        for batch in dataloader:
            fieldpatch  = batch['fieldpatch'].to(device)
            quadpatch   = batch['quadpatch'].to(device)
            localvalues = batch['localvalues'].to(device) if uselocal else None
            if hasattr(model,'kernellayer'):
                outputvalues,features = model(fieldpatch,quadpatch,localvalues)
                featureslist.append(features.cpu().numpy())
            else:
                outputvalues = model(fieldpatch,localvalues)
            outputslist.append(outputvalues.cpu().numpy())
    predictions = np.concatenate(outputslist,axis=0)
    features    = np.concatenate(featureslist,axis=0) if featureslist else None
    return predictions,features


def scatter(data,kind,*,centers=None,refda=None,nkernels=None,kerneldims=None,nonparam=None,fieldvars=FIELDVARS):
    """
    Purpose
    -------
    Scatter patched outputs (predictions/features) onto a full ref grid, or slice kernel weights
    down to the requested kerneldims, returning numpy array(s) plus xarray dimension metadata.

    Args
    ----
    data (np.ndarray):
        Model output: predictions, integrated features, or weights.
    kind (str):
        One of {'predictions','features','weights'}.
    centers (list[tuple[int,int,int]] | None):
        Patch centers (latidx, lonidx, timeidx). Required for predictions/features.
    refda (xr.DataArray | None):
        Reference grid/coords (lat, lon, time, ...). Required for predictions/features.
    nkernels (int | None):
        Number of kernels/members. Required for features; required for ensemble predictions.
    kerneldims (list[str] | tuple[str] | None):
        Kernel-varying dims (subset of {'lat','lon','lev','time'}). Required for weights.
    nonparam (bool | None):
        Whether kernel is nonparametric.
    fieldvars (list[str]):
        Predictor names for features/weights.

    Returns
    -------
    tuple[dict, dict]:
        spec, attrs

        spec:
          - "arrays": dict[str, np.ndarray]  (varname -> array)
          - "dims":   dict[str, tuple[str,...]]
          - "coords": dict[str, dict]
        attrs:
          - dict[str, dict] (varname -> attrs dict)
    """
    if kind not in {"predictions", "features", "weights"}:
        raise ValueError("`kind` must be one of {'predictions','features','weights'}")

    spec  = {'arrs':{},'dims':{},'coords':{}}
    attrs = {}

    if kind in {"predictions", "features"}:
        if refda is None or centers is None:
            raise ValueError("`refda` and `centers` required for prediction/feature reformatting")
        latidxs, lonidxs, timeidxs = map(np.asarray, zip(*centers))

    if kind == "predictions":
        isensemble = bool(nonparam) and (nkernels is not None) and data.ndim == 2 and data.shape[1] == nkernels
        if isensemble:
            arr = np.full((nkernels, *refda.shape), np.nan, dtype=refda.dtype)
            arr[:, latidxs, lonidxs, timeidxs] = data.T
            dims = ("member",) + refda.dims
            coords = {"member": np.arange(nkernels), **refda.coords}
        else:
            arr = np.full(refda.shape, np.nan, dtype=refda.dtype)
            arr[latidxs, lonidxs, timeidxs] = data
            dims = refda.dims
            coords = dict(refda.coords)

        spec["arrays"]["pr"] = arr
        spec["dims"]["pr"] = dims
        spec["coords"]["pr"] = coords
        attrs["pr"] = {
            "long_name": "Predicted precipitation rate (log1p-transformed and standardized)",
            "units": "N/A",
        }
        return spec, attrs

    if kind == "features":
        if nkernels is None:
            raise ValueError("`nkernels` required for feature reformatting")
        if data.shape[1] != len(fieldvars) * nkernels:
            raise ValueError("`data.shape[1]` must equal len(fieldvars) × nkernels")

        vals = data.reshape(len(centers), len(fieldvars), nkernels)  # (center, field, member)

        if nonparam:
            # arr: (member, field, ...grid...)
            arr = np.full((nkernels, len(fieldvars), *refda.shape), np.nan, dtype=refda.dtype)
            arr[:, :, latidxs, lonidxs, timeidxs] = vals.transpose(2, 1, 0)

            dims = ("member",) + refda.dims
            coords = {"member": np.arange(nkernels), **refda.coords}

            for i, varname in enumerate(fieldvars):
                spec["arrays"][varname] = arr[:, i, ...]
                spec["dims"][varname] = dims
                spec["coords"][varname] = coords
                attrs[varname] = {
                    "long_name": f"{varname} (kernel-integrated and standardized)",
                    "units": "N/A",
                }
        else:
            # deterministic => use member/kernel 0 only
            arr = np.full((len(fieldvars), *refda.shape), np.nan, dtype=refda.dtype)
            arr[:, latidxs, lonidxs, timeidxs] = vals[:, :, 0].T  # (field, center) -> (field, lat, lon, time)

            dims = refda.dims
            coords = dict(refda.coords)

            for i, varname in enumerate(fieldvars):
                spec["arrays"][varname] = arr[i, ...]
                spec["dims"][varname] = dims
                spec["coords"][varname] = coords
                attrs[varname] = {
                    "long_name": f"{varname} (kernel-integrated and standardized)",
                    "units": "N/A",
                }

        return spec, attrs


    if kerneldims is None:
        raise ValueError("`kerneldims` required for weight reformatting")

    kerneldims = tuple(kerneldims)
    base = ("lat", "lon", "lev", "time")
    kept = tuple(d for d in base if d in kerneldims)

    if nonparam:
        # expected (field, member, lat, lon, lev, time)
        alldims = ("field", "member") + base
        if data.ndim != len(alldims):
            raise ValueError(f"Nonparam weights expected {len(alldims)}D array shaped like {alldims}")

        indexer = tuple(
            slice(None) if (d in ("field", "member") or d in kerneldims) else 0
            for d in alldims
        )
        weights = data[indexer]
        dims = ("field", "member") + kept
        coords = {"field": fieldvars, "member": np.arange(weights.shape[1])}
        for ax, d in enumerate(dims[2:], start=2):
            coords[d] = np.arange(weights.shape[ax])
    else:
        # expected (field, lat, lon, lev, time); if (field, nkernels, ...) drop nkernels axis
        if data.ndim == 6:
            data = data[:, 0, ...]
        alldims = ("field",) + base
        if data.ndim != len(alldims):
            raise ValueError(f"Parametric weights expected {len(alldims)}D array shaped like {alldims}")

        indexer = tuple(slice(None) if (d == "field" or d in kerneldims) else 0 for d in alldims)
        weights = data[indexer]
        dims = ("field",) + kept
        coords = {"field": fieldvars}
        for ax, d in enumerate(dims[1:], start=1):
            coords[d] = np.arange(weights.shape[ax])

    spec["arrays"]["weights"] = weights
    spec["dims"]["weights"] = dims
    spec["coords"]["weights"] = coords
    attrs["weights"] = {
        "long_name": "Nonparametric kernel weights" if nonparam else "Parametric kernel weights",
        "units": "N/A",
    }
    return spec, attrs


def dataset(spec attrs):
    """
    Purpose
    -------
    Build an xr.Dataset from scatter() output.

    Args
    ----
    spec (dict):
        Output from scatter(): dict with keys {"arrays","dims","coords"}.
        Each is a mapping varname -> array/dims/coords.
    attrs (dict[str, dict]):
        Mapping varname -> attrs dict (e.g., {"long_name": ..., "units": ...}).

    Returns
    -------
    xr.Dataset
        Dataset containing each variable as an xr.DataArray with the requested dims/coords/attrs.
    """
    ds = xr.Dataset()
    for name, arr in spec["arrays"].items():
        da = xr.DataArray(arr, dims=spec["dims"][name], coords=spec["coords"][name], name=name)
        da.attrs = attrs.get(name, {})
        ds[name] = da
    return ds







def reformat(data,kind,*,centers=None,refda=None,nkernels=None,kerneldims=None,nonparam=False,fieldvars=FIELDVARS):
    '''
    Purpose: Reformat predictions, features, or weights into xr.Dataset objects with consistent dimensions.
    Args:
    - data (np.ndarray): predictions, kernel-integrated features, or kernel weights output by inference
    - kind (str): 'predictions' | 'features' | 'weights'}
    - centers (list[tuple[int,int,int]] | None): list of (latidx, lonidx, timeidx) patch centers
    - refda (xr.DataArray | None): reference DataArray for reconstructing predictions and features or None
    - nkernels (int | None): number of learned kernels or None
    - kerneldims (list[str] | tuple[str] | None): dimensions along which the kernel varies for reconstructing weights or None
    - nonparam (bool): whether the kernel is non-parametric
    - fieldvars (list[str]): predictor variable names (defaults to FIELDVARS)
    Returns:
    - xr.Dataset: Dataset of predictions, features, or weights
    '''
    if kind=='predictions':
        if refda is None or centers is None:
            raise ValueError('`refda` and `centers` required for prediction reformatting')
        nlats,nlons,ntimes = refda.shape 
        if nonparam and data.ndim==2 and data.shape[1]==nkernels:
            arr = np.full((nkernels,nlats,nlons,ntimes),np.nan,dtype=np.float32)
            for i,(latidx,lonidx,timeidx) in enumerate(centers):
                arr[:,latidx,lonidx,timidx] = data[i]
            da = xr.DataArray(arr,dims=('member',)+refda.dims,coords={'member':np.arange(nkernels),**refda.coords},name='pr')
        else:
            arr = np.full(refda.shape,np.nan,dtype=np.float32)
            for i,(latidx,lonidx,timeidx) in enumerate(centers):
                arr[latidx,lonidx,timeidx] = data[i]
            da = xr.DataArray(arr,dims=refda.dims,coords=refda.coords,name='pr')
        da.attrs = dict(long_name='Predicted precipitation rate (log1p-transformed and standardized)',units='N/A')
        return da.to_dataset()
    elif kind=='features':
        if refda is None or centers is None:
            raise ValueError('`refda` and `centers` required for feature reformatting')
        if data.shape[1]!=len(fieldvars)*nkernels:
            raise ValueError('`data.shape[1]` must equal len(fieldvars) × nkernels')
        nsamples,nfeatures = data.shape
        nlats,nlons,ntimes = refda.shape
        data = data.reshape(nsamples,len(fieldvars),nkernels)
        ds   = xr.Dataset()
        if nonparam:
            arr = np.full((nkernels,len(fieldvars),nlats,nlons,ntimes),np.nan,dtype=np.float32)
            for i,(latidx,lonidx,timidx) in enumerate(centers):
                arr[:,:,latidx,lonidx,timidx] = data[i].transpose(1,0)
            for fieldidx,varname in enumerate(fieldvars):
                da = xr.DataArray(arr[:,fieldidx,...],dims=('member',)+refda.dims,coords={'member':np.arange(nkernels),**refda.coords},name=varname)
                da.attrs = dict(long_name=f'{varname} (kernel-integrated and standardized)',units='N/A')
                ds[da.name] = da
        else:
            data = data[...,0]
            arr  = np.full((len(fieldvars),nlats,nlons,ntimes),np.nan,dtype=np.float32)
            for i,(latidx,lonidx,timidx) in enumerate(centers):
                arr[:,latidx,lonidx,timidx] = data[i]
            for fieldidx,varname in enumerate(fieldvars):
                da = xr.DataArray(arr[fieldidx,...],refda.dims,coords=refda.coords,name=varname)
                da.attrs = dict(long_name=f'{varname} (kernel-integrated and standardized)',units='N/A')
                ds[da.name] = da
        return ds    
    elif kind=='weights':
        if kerneldims is None:
            raise ValueError('`kerneldims` required for weight reformatting')
        kerneldims = tuple(kerneldims)
        alldims    = ['field','member','lat','lon','lev','time']
        if nonparam:
            indexer = [slice(None) if dim in ('field','member') or dim in kerneldims else 0 for dim in alldims]
            weights = data[tuple(indexer)]
            dims    = ['field','member']+[dim for dim in ('lat','lon','lev','time') if dim in kerneldims]
            coords  = {'field':fieldvars,'member':np.arange(data.shape[1])}
            for ax,dim in enumerate(dims[2:],start=2):
                coords[dim] = np.arange(weights.shape[ax])
            da = xr.DataArray(weights,dims=dims,coords=coords,name='weights')
            da.attrs = dict(long_name='Nonparametric kernel weights',units='N/A')
            return da.to_dataset()
        else:
            nfieldvars,nkernels,plats,plons,plevs,ptimes = data.shape
            indexer = [slice(None) if dim=='field' or dim in kerneldims else 0 for dim in alldims]
            weights = data[tuple(indexer)]
            dims    = ['field']+[dim for dim in ('lat','lon','lev','time') if dim in kerneldims]
            coords  = {'field':fieldvars}
            for ax,dim in enumerate(dims[1:],start=1):
                coords[dim] = np.arange(weights.shape[ax])
            da = xr.DataArray(weights,dims=dims,coords=coords,name='weights')
            da.attrs = dict(long_name=f'Parametric kernel weights',units='N/A')
            return da.to_dataset()

def save(name,ds,kind,split,savedir):
    '''
    Purpose: Save an xr.Dataset of prediction, features, or weights to NetCDF and verify by reopening.
    Args:
    - name (str): model name
    - ds (xr.Dataset): Dataset containing predictions, kernel-integrated features, or kernel weights
    - kind (str): 'predictions' | 'features' | 'weights'
    - split (str): 'valid' | 'test'
    - resultsdir (str): output directory
    Returns:
    - bool: True if save and verification successful, False otherwise
    '''
    os.makedirs(savedir,exist_ok=True)
    filename = f'{name}_{split}_{kind}.nc'
    filepath = os.path.join(savedir,filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        ds.to_netcdf(filepath,engine='h5netcdf')
        with xr.open_dataset(filepath,engine='h5netcdf') as _:
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False
        
def parse():
    '''
    Purpose: Parse command-line arguments for running the evaluation script.
    Returns:
    - argparse.Namespace: parsed arguments
    '''
    parser = argparse.ArgumentParser(description='Evaluate NN precipitation models.')
    parser.add_argument('--models',type=str,default='all',help='Comma-separated list of model names to evaluate, or `all`.')
    parser.add_argument('--split',type=str,required=True,choices=['valid','test'],help='Which split to evaluate (`valid` or `test`).')
    return parser.parse_args()
    
if __name__=='__main__':
    logger.info('Setting random seed...')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    logger.info('Determining device type...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    logger.info('Parsing arguments...')
    args   = parse()
    models = [m.strip() for m in args.models.split(',')] if args.models!='all' else None
    split  = args.split
    logger.info('Preparing evaluation split...')
    splitdata = DataModule.prepare([split],FIELDVARS,LOCALVARS,TARGETVAR,SPLITDIR)
    cachedconfig = None
    cachedresult = None
    for modelconfig in config.models:
        name = modelconfig['name']
        if models is not None and name not in models:
            continue
        logger.info(f'Running {name}...')
        patchconfig   = modelconfig['patch']
        uselocal      = modelconfig['uselocal']
        currentconfig = (patchconfig['radius'],patchconfig['maxlevs'],patchconfig['timelag'],uselocal)
        if currentconfig==cachedconfig:
            logger.info('   Reusing cached datasets and loaders...')
            result = cachedresult
        else:
            logger.info('   Building new datasets and loaders....')
            result = DataModule.dataloaders(splitdata,patchconfig,uselocal,LATRANGE,LONRANGE,BATCHSIZE,WORKERS,device)
            cachedconfig = currentconfig
            cachedresult = result
        logger.info('   Initializing model and populating trained weights....')
        model = load(name,modelconfig,result,device)
        logger.info('   Starting inference....')
        predictions,features = inference(model,split,result,uselocal,device)
        logger.info('Saving outputs...')
        centers   = result['centers'][split]
        refda     = splitdata[split]['ds'][TARGETVAR]
        haskernel = hasattr(model,'kernellayer')
        nonparam  = haskernel and isinstance(model.kernellayer,NonparametricKernelLayer)
        nkernels  = model.kernellayer.nkernels if haskernel else 1
        ds = reformat(predictions,kind='predictions',centers=centers,refda=refda,nkernels=nkernels,nonparam=nonparam)
        save(name,ds,'predictions',split,PREDSDIR)
        if haskernel:
            weights = model.kernellayer.weights(result['quad'],device,asarray=True)
            ds = reformat(weights,kind='weights',nkernels=nkernels,kerneldims=model.kernellayer.kerneldims,nonparam=nonparam)
            save(name,ds,'weights',split,WEIGHTSDIR)
            if features is not None:
                ds = reformat(features,'features',centers=centers,refda=refda,nkernels=nkernels,nonparam=nonparam)
                save(name,ds,'features',split,FEATSDIR)
        del model,predictions,features,centers,refda,haskernel,nonparam,nkernels,ds