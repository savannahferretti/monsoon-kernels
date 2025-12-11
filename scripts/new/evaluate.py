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
SPLITDIR       = config.splitdir
MODELDIR       = config.modeldir
PREDICTIONSDIR = config.predictionsdir
FEATUREDIR     = config.featuredir
FIELDVARS      = config.fieldvars
LOCALVARS      = config.localvars
TARGETVAR      = config.targetvar
LATRANGE       = config.latrange
LONRANGE       = config.lonrange
SEED           = config.seed
WORKERS        = config.workers
BATCHSIZE      = config.batchsize

def load(name,nfieldvars,ncolcalvars,modelconfig,modeldir=MODELDIR):
    '''
    Purpose: Load a trained model from checkpoint.
    Args:
    - name (str): model name
    - nfieldvars (int): number of predictor fields
    - patchshape (?): (plats, plons, plevs, ptimes)
    - nlocalvars (int): number of local inputs
    - modelconfig (dict): model configuration
    - modeldir (str): directory containing checkpoints (defaults to MODELDIR)
    Returns:
    - torch.nn.Module: model with loaded state_dict or None if checkpoint not found
    '''
    filename = f'{name}.pth'
    filepath = os.path.join(modeldir,filename)
    if not os.path.exists(filepath):
        logger.error(f'   Checkpoint not found: {filepath}')
        return None
    model = build(name,modelconfig,nfieldvars,patchshape,nlocalvars)
    state = torch.load(filepath,map_location='cpu')
    model.load_state_dict(state)
    return model

def inference(model,dataloader,centers,quad,refdadevice):
    '''
    Purpose: Run model, denormalize, and reconstruct full (lat,lon,time) field.
    Args:
    - model (torch.nn.Module): trained model
    - dataloader (DataLoader): data loader for evaluation
    - centers (list[tuple[int,int,int]]): list of (latidx, lonidx, timeidx) patch centers
    - quad (torch.Tensor): quadrature weights for kernel models
    - device (str): device to use
    Returns:
    - xr.DataArray: predicted precipitation DataArray
    '''
    model = model.to(device)
    quadweights = quadweights.to(device) if quadweights is not None else None
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in dataloader:
            patch = batch['patch'].to(device)
            local = batch['local'].to(device) if uselocal else None
            output = model(patch,local,quad) if quad is not None else model(patch,local)
            outputs.append(output.cpu().numpy())
    predictions = np.concatenate(outputs,axis=0)
    arr = np.full(refda.shape,np.nan,dtype=np.float32)
    for i,(latidx,lonidx,timeidx) in enumerate(centers):
        arr[latidx,lonidx,timeidx] = predictions[i]
    da = xr.DataArray(arr,dims=refda.dims,coords=refda.coords,name='pr')
    da.attrs = dict(long_name='NN-predicted precipitation rate',units='N/A')
    return da

def save(da,name,split,savedir=PREDICTIONSDIR):
    '''
    Purpose: Save predicted precipitation xr.DataArray to NetCDF and verify by reopening.
    Args:
    - da (xr.DataArray): prediction DataArray
    - name (str): model name
    - split (str): 'valid' | 'test'
    - resultsdir (str): output directory (defaults to PREDICTIONSDIR)
    Returns:
    - bool: True if save and verification successful, False otherwise
    '''
    os.makedirs(savedir,exist_ok=True)
    filename = f'{name}_{split}_pr.nc'
    filepath = os.path.join(savedir,filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        da.to_netcdf(filepath,engine='h5netcdf')
        with xr.open_dataset(filepath,engine='h5netcdf') as _:
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

    
    def extract(self,item,name,savedir):
        '''
        Purpose: Save kernel weights or kernel-integrated features to a NumPy zip archive then verify by reopening.
        Args:
        - item (np.ndarray | dict): either kernel-integrated features with shape (nsamples, nfieldvars*nkernels)
          or dictionary returned by weights()
        - name (str): model name
        - savedir (str): output directory
        Returns:
        - bool: True if save and verification successful, False otherwise
        '''
        os.makedirs(savedir,exist_ok=True)
        filename = f'{name}_kernel_features.npz' if isinstance(item,np.ndarray) else f'{name}_kernel_weights.npz'
        filepath = os.path.join(savedir,filename)
        logger.info(f'   Attempting to save {filename}...')
        try:
            if isinstance(item,np.ndarray):
                np.savez(filepath,item)
            else:
                arrs = {}
                for key,value in item.items():
                    if isinstance(value,dict):
                        for subkey,subvalue in value.items():
                            arrs[f'{key}_{subkey}'] = subvalue
                    else:
                        arrs[key] = value
                np.savez(filepath,**arrs)
            with np.load(filepath) as _:
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
    splitdata    = DataModule.prepare(['train','valid'],FIELDVARS,LOCALVARS,TARGETVAR,SPLITDIR)
    cachedconfig = None
    cachedresult = None
    for name,modelconfig in config.models.items():
        name = modelconfig['name']
        if name not in models:
            continue
        logger.info(f'Evaluating {name}...')
        patchconfig = modelconfig['patch']
        uselocal    = modelconfig['uselocal']
        logger.info('   Building data loaders....')
        currentconfig = (patchconfig['radius'],patchconfig['maxlevs'],patchconfig['timelag'],uselocal)
        if currentconfig==cachedconfig:
            logger.info('   Reusing cached datasets and loaders...')
            result = cachedresult
        else:
            result = DataModule.dataloaders(splitdata,patchconfig,uselocal,LATRANGE,LONRANGE,BATCHSIZE,WORKERS,device)
            cachedconfig = currentconfig
            cachedresult = result
        logger.info('   Initializing model....')
        nfieldvars = len(NFIELDVARS)
        nlocalvars = len(LOCALVARS)        
        model = load(name,nfieldvars,nlocalvars).to(device)
        logger.info('   Starting inference....')
        evalloader = data['loaders'][split]
        centers    = result['centers'][split]
        target     = splitdata[split]['target']
        quad       = data['quad']       
        refda = splitdata[split]['ds'][TARGETVAR]
        arr = inference(model,evalloader,centers,target,quad,device)
        save(arr,refda,name,split)
        logger.info('Extracting weights and features...')
        weights = KernelNN.weights()
        save(weights,name)
        features = 
        save(features,name)
        del model,arr,