#!/usr/bin/env python

import os
import torch
import wandb
import logging
import argparse
import numpy as np
import xarray as xr
from utils import Config
from dataset import DataModule
from models import BaselineNN,KernelNN,ModelFactory
from kernels import NonparametricKernelLayer,ParametricKernelLayer

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
    Purpose: Initialize a model instance from ModelFactory.build() and populate with weights from from a saved checkpoint.
    Args:
    - name (str): model name
    - modelconfig (dict): model configuration
    - result (dict[str,object]): dictionary from DataModule.dataloaders()
    - device (str): device to use ('cpu' | 'cuda')
    - fieldvars (int): predictor field variable names (defaults to FIELDVARS)
    - localvars (int): local input variable names (defaults to LOCALVARS)
    - modeldir (str): directory containing checkpoints (defaults to MODELDIR)
    Returns:
    - torch.nn.Module: model with loaded state_dict on 'device' or None if checkpoint not found
    '''
    filedir  = os.path.join(modeldir,kind)
    filename = f'{name}.pth'
    filepath = os.path.join(filedir,filename)
    if not os.path.exists(filepath):
        logger.error(f'   Checkpoint not found: {filepath}')
        return None
    patchshape = result['geometry'].shape
    nfieldvars = len(fieldvars)
    nlocalvars = len(localvars)
    model = ModelFactory.build(name,modelconfig,patchshape,nfieldvars,nlocalvars)
    state = torch.load(filepath,map_location='cpu')
    model.load_state_dict(state)
    return model.to(device)

def inference(model,result,uselocal,device):
    '''
    Purpose: Run inference on the model.
    Args:
    - model (torch.nn.Module): trained model instance 
    - result (dict[str,object]): dictionary from DataModule.dataloaders()
    - uselocal (bool): whether to use local inputs
    - device (str): device to use
    Returns:
    - xr.DataArray: predicted precipitation DataArray
    '''
    dataloader = result['loader']
    model.eval()
    outputslist  = []
    featureslist = []
    with torch.no_grad():
        for batch in dataloader:
            patch = batch['patch'].to(device)
            local = batch['local'].to(device) if uselocal else None
            if hasattr(model,'kernellayer'):
                quad = result['quad'].to(device)
                output,feature = model(patch,quad,local,returnfeatures=True)
                featureslist.append(feature.cpu().numpy())
            else:
                output = model(patch,local)
            outputslist.append(output.cpu().numpy())
    predictions = np.concatenate(outputslist,axis=0)
    features    = np.concatenate(featureslist,axis=0) if featureslist else None
    return predictions,features

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
    - nonparam (bool): True for non-parametric kernels, False for parametric kernels
    - fieldvars (list[str]): predictor variable names(defaults to FIELDVARS)
    Returns:
    - xr.Dataset: Datset of predictions, features, or weights
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
                da.attrs(long_name=f'{varname} (kernel-integrated and standardized)',units='N/A')
                ds[da.name] = da
        else:
            data = data[...,0]
            arr  = np.full((len(fieldvars),nlats,nlons,ntimes),np.nan,dtype=np.float32)
            for i,(latidx,lonidx,timidx) in enumerate(centers):
                arr[:,latidx,lonidx,timidx] = data[i]
            for fieldidx,varname in enumerate(fieldvars):
                da = xr.DataArray(arr[fieldidx,...],refda.dims,coords=refda.coords,name=varname)
                da.attrs(long_name=f'{varname} (kernel-integrated and standardized)',units='N/A')
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
            da.attrs = dict(long_name='Nonparametric kernel weights',units='0-1')
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
            da.attrs = dict(long_name=f'Parametric kernel weights',units='0-1')
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
    for name,modelconfig in config.models.items():
        name = modelconfig['name']
        if name not in models:
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
        model = load(name,modelconfig,result,uselocal,device)
        logger.info('   Starting inference....')
        predictions,features = inference(model,result,device)
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