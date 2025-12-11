#!/usr/bin/env python

import os
import torch
import logging
import argparse
import numpy as np
import xarray as xr
from utils import Config
from dataset import DataPrep
from datetime import datetime
from network import NonparametricKernelLayer,ParametricKernelLayer,BaselineNN,KernelNN

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

config = Config()
SPLITDIR       = config.splitdir
MODELDIR       = config.modeldir
PREDICTIONDIR  = config.predictiondir
FEATUREDIR     = config.featuredir
FIELDVARS      = config.fieldvars
LOCALVARS      = config.localvars
TARGETVAR      = config.targetvar
LATRANGE       = config.latrange
LONRANGE       = config.lonrange
WORKERS        = config.workers
BATCHSIZE      = config.batchsize
CRITERION      = config.criterion

def load(name,modeldir=MODELDIR):
    '''
    Purpose: Load a trained model.
    Args:
    - model (torch.nn.Module): trained model instance
    - name (str): model name
    - modeldir (str): directory containing checkpoints (defaults to MODELDIR)
    Returns:
    - bool: True if checkpoint loaded successfully, False otherwise
    '''
    filename = f'{name}.pth'
    filepath = os.path.join(modeldir,filename)
    if not os.path.exists(filepath):
        logger.error(f'   Checkpoint not found: {filepath}')
        return False
    state = torch.load(filepath,map_location='cpu')
    model.load_state_dict(state)
    logger.info(f'   Loaded checkpoint from {filename}')
    return True

def inference(model,loader,centers,targettemplate,quadweights,device=DEVICE,ampeneabled=AMPENABLED):
    '''
    Purpose: Run model, denormalize, and reconstruct full (lat,lon,time) field.
    Args:
    - model (torch.nn.Module): trained model
    - loader (DataLoader): data loader for evaluation
    - centers (list[tuple[int,int,int]]): (latidx, lonidx, timeidx) for each sample
    - targettemplate (torch.Tensor): (nlats, nlons, ntimes) template for reconstruction
    - quadweights (torch.Tensor): quadrature weights for kernel models
    - device (str): device to run inference on (defaults to DEVICE)
    Returns:
    - np.ndarray: reconstructed precipitation field (nlats, nlons, ntimes)
    '''
    model = model.to(device)
    quadweights = quadweights.to(device) if quadweights is not None else None
    model.eval()
    outputs = []
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=ampenabled):
            for batch in loader:
                patch = batch['patch'].to(device)
                local = batch.get('local').to(device) if uselocal else None
                if hasattr(model,'kernellayer'):
                    output = model(patch,local,quadweights)
                else:
                    output = model(patch,local)
                outputs.append(output.cpu().numpy())
    predictions = np.concatenate(outputs,axis=0)
    arr = np.full(targettemplate.shape,np.nan,dtype=np.float64)
    for i,(latidx,lonidx,timeidx) in enumerate(centers):
        arr[latidx,lonidx,timeidx] = predictions[i]
    return arr


def save(arr,refda,name,split,savedir=PREDICTIONDIR):
    '''
    Purpose: Save predicted precipitation to NetCDF on the target grid.
    Args:
    - arr (np.ndarray): precipitation predictions (nlats, nlons, ntimes)
    - refda (xr.DataArray): reference DataArray with correct coordinates/dims
    - name (str): model name
    - split (str): 'valid' | 'test'
    - resultsdir (str): output directory (defaults to PREDICTIONDIR)
    Returns:
    - bool: True if save and verification successful, False otherwise
    '''
    os.makedirs(savedir,exist_ok=True)
    da = xr.DataArray(arr,dims=refda.dims,coords=refda.coords,name='pr')
    da.attrs = dict(long_name='NN-predicted precipitation rate',units='N/A')
    filename = f'{name}_{split}_pr.nc'
    filepath = os.path.join(savedir,filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        da.to_netcdf(filepath, engine='h5netcdf')
        with xr.open_dataset(filepath, engine='h5netcdf') as _:
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False



    def save(self,item,name,savedir):
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
    
if __name__ == '__main__':
    logger.info('Setting random seed...')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    logger.info('Determining device type...')
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
    AMPENABLED = (DEVICE=='cuda')
    if AMPENABLED:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    logger.info('Parsing arguments...')
    args   = parse()
    models = [m.strip() for m in args.models.split(',')] if args.models!='all' else None
    split  = args.split
    logger.info(f'Loading {split if split=='test' else 'validation'} split...')
    splitdata = DataPrep.prepare(FILEDIR,FIELDVARS,LOCALVARS,TARGETVAR,[split])
    cachedconfig = None
    cachedresult = None
    for modelconfig in MODELCONFIGS:
        name = modelconfig['name']
        if (models is not None) and (name not in requested):
            continue
        logger.info(f'Evaluating {name}...')
        patchconfig = modelconfig['patch']
        uselocal    = modelconfig.get('uselocal',True)
        currentconfig = (patchconfig['radius'],patchconfig['maxlevs'],patchconfig['timelag'],uselocal)
        if currentconfig==cachedconfig:
            logger.info('   Reusing cached dataset and loader')
            result = cachedresult
        else:
            result = DataPrep.dataloaders(splitdata,patchconfig,uselocal,LATRANGE,LONRANGE,BATCHSIZE,WORKERS,DEVICE,splitdata[split]['ds'].lev.size)
            cachedconfig = currentconfig
            cachedresult = result
        model = load(name)
        arr = inference(model,result['loaders'][split],result['centers'][split],splitdata[split]['target'],result['quadweights'],DEVICE)
        save(arr,splitdata[split]['ds'][TARGETVAR],name,split)