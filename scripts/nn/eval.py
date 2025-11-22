#!/usr/bin/env python

import os
import glob
import json
import torch
import logging
import argparse
import warnings
import numpy as np
import xarray as xr
from model import NNModel
from torch.utils.data import TensorDataset,DataLoader

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

with open('configs.json','r',encoding='utf8') as f:
    CONFIGS = json.load(f)
FILEDIR     = CONFIGS['paths']['filedir']
MODELDIR    = CONFIGS['paths']['modeldir']
RESULTSDIR  = CONFIGS['paths']['resultsdir']
TARGETVAR   = CONFIGS['params']['targetvar']
BATCHSIZE   = CONFIGS['params']['batchsize']
RUNS        = CONFIGS['runs']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def reshape(da):
    '''
    Purpose: Convert an xr.DataArray into a 2D NumPy array suitable for NN I/O.
    Args:
    - da (xr.DataArray): 3D or 4D DataArray
    Returns:
    - np.ndarray: shape (nsamples, nfeatures); for 3D, nfeatures=1, for 4D, nfeatures equals the size of the 'lev' dimension
    '''
    if 'lev' in da.dims:
        da  = da.sortby('lev',ascending=False) 
        arr = da.transpose('time','lat','lon','lev').values.reshape(-1,da.lev.size)
    else:
        arr = da.transpose('time','lat','lon').values.reshape(-1,1)
    return arr
    
def load(splitname,inputvars,targetvar=TARGETVAR,filedir=FILEDIR):
    '''
    Purpose: Load in a normalized validation or test split and build a 2D feature matrix for the NN. 
    Args:
    - splitname (str): 'normvalid' | 'normtest'
    - inputvars (list[str]): list of input variables
    - targetvar (str): target variable name (defaults to TARGETVAR)
    - filedir (str): directory containing split files (defaults to FILEDIR)
    Returns:
    - tuple[torch.FloatTensor,xr.DataArray]: 2D input tensor and target DataArray (for reshaping predictions)
    '''
    if splitname not in ('normvalid','normtest'):
        raise ValueError('Splitname must be `normvalid` or `normtest`.')
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    varlist  = list(inputvars)+[targetvar]
    ds = xr.open_dataset(filepath,engine='h5netcdf')[varlist]
    Xlist = [reshape(ds[inputvar]) for inputvar in inputvars]
    X = np.concatenate(Xlist,axis=1) if len(Xlist)>1 else Xlist[0]
    X = torch.tensor(X,dtype=torch.float32)
    ytemplate = ds[targetvar]
    return X,ytemplate

def get_checkpoints(runname,modeldir=MODELDIR):
    '''
    Purpose: Return a sorted list of checkpoint filepaths for a given model run.
    Args:
    - runname (str): model run name
    - modeldir (str): directory with saved model checkpoints (defaults to MODELDIR)
    Returns:
    - list[str]: list of checkpoint filepaths matching the run name
    '''
    pattern     = os.path.join(modeldir,f'nn_{runname}_[0-9][0-9].pth')
    checkpoints = sorted(glob.glob(pattern))
    return checkpoints

def fetch(checkpoint,inputsize,device=DEVICE):
    '''
    Purpose: Rebuild a trained NN model from a specific checkpoint file.
    Args:
    - checkpoint (str): full filepath to a saved checkpoint file
    - inputsize (int): number of input features to initialize NNModel
    - device (str): 'cuda' or 'cpu' device for model evaluation (defaults to DEVICE)
    Returns:
    - NNModel: model on 'device' with loaded state_dict (weights)
    '''
    model = NNModel(inputsize).to(device)
    state = torch.load(checkpoint,map_location=device)
    model.load_state_dict(state)
    return model

def denormalize(ynormflat,targetvar=TARGETVAR,filedir=FILEDIR):
    '''
    Purpose: Convert normalized precipitation predictions back to physical units by undoing z-score normalization and log1p transformation. 
    Args:
    - ynormflat (np.ndarray): vector of normalized predictions
    - targetvar (str): target variable name (defaults to TARGETVAR)
    - filedir (str): directory where JSON file is stored (defaults to FILEDIR)
    Returns:
    - np.ndarray: vector of denormalized predictions in original units
    '''
    with open(os.path.join(filedir,'stats.json'),'r',encoding='utf-8') as f:
        stats = json.load(f)
    mean = float(stats[f'{targetvar}_mean'])
    std  = float(stats[f'{targetvar}_std'])
    ylog = ynormflat*std+mean
    y = np.expm1(ylog)
    return y

def predict(model,X,ytemplate,batchsize=BATCHSIZE,device=DEVICE):
    '''
    Purpose: Run the NN forward pass in batches and return precipitation predictions as an xr.DataArray.
    Args:
    - model (NNModel): trained/loaded NN model
    - X (torch.Tensor): 2D input tensor
    - ytemplate (xr.DataArray): template with dimensions/coordinates to reshape predictions
    - batchsize (int): inference batch size (defaults to BATCHSIZE)
    - device (str): 'cuda' or 'cpu' device for model evaluation (defaults to DEVICE)
    Returns:
    - xr.DataArray: 3D DataArray of predicted precipitation 
    '''
    evaldataset = TensorDataset(X)
    evalloader  = DataLoader(evaldataset,batch_size=batchsize,shuffle=False,pin_memory=True)
    ypredlist = []
    model.eval()
    with torch.no_grad():
        for (Xbatch,) in evalloader:
            Xbatch = Xbatch.to(device,non_blocking=True)
            ybatchpred = model(Xbatch)
            ypredlist.append(ybatchpred.squeeze(-1).cpu().numpy())
    ynormflat = np.concatenate(ypredlist,axis=0)
    ypredflat = denormalize(ynormflat)
    da = xr.DataArray(ypredflat.reshape(ytemplate.shape),dims=ytemplate.dims,coords=ytemplate.coords,name='pr')
    da.attrs = dict(long_name='NN-predicted precipitation rate',units='mm/hr')
    return da

def save(ypred,runname,splitname,resultsdir=RESULTSDIR):
    '''
    Purpose: Save an xr.Dataset of an ensemble of predicted precipitation to a NetCDF file, then verify the write by reopening.
    Args:
    - ypred (xr.Dataset): Dataset of predicted precipitation with a 'member' dimension
    - runname (str): model run name (or run name plus suffix indicating statistic)
    - splitname (str): evaluated split label
    - resultsdir (str): output directory (defaults to RESULTSDIR)
    Returns:
    - bool: True if writing and verification succeed, otherwise False
    '''
    os.makedirs(resultsdir,exist_ok=True)
    filename = f'nn_{runname}_{splitname}_pr.nc'
    filepath = os.path.join(resultsdir,filename)
    logger.info(f'      Attempting to save {filename}...')
    try:
        ypred.to_netcdf(filepath,engine='h5netcdf')
        with xr.open_dataset(filepath,engine='h5netcdf') as _:
            pass
        logger.info('         File write successful')
        return True
    except Exception:
        logger.exception('         Failed to save or verify')
        return False

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate NN models on a chosen split.')
    parser.add_argument('--split',required=True,choices=['normvalid','normtest'],help='Which split to evaluate: `normvalid` or `normtest`.')
    args = parser.parse_args()
    logger.info(f'Evaluating NN models on {args.split} set...')
    for run in RUNS:
        runname     = run['run_name']
        inputvars   = run['input_vars']
        description = run['description']
        logger.info(f'   Evaluating {runname}')
        X,ytemplate = load(args.split,inputvars)
        checkpoints = get_checkpoints(runname)
        memberpreds = []
        for checkpoint in checkpoints:
            model = fetch(checkpoint,X.shape[1])
            ypred = predict(model,X,ytemplate)
            memberpreds.append(ypred)
            del model,ypred
        ensemble = xr.concat(memberpreds,dim='member')
        ensemble = ensemble.assign_coords(member=np.arange(len(memberpreds)))
        ensemble.name = 'pr'
        ensemble.attrs = dict(long_name='Ensemble NN-predicted precipitation rate',units='mm/hr')
        save(ensemble,runname,args.split)
        del X,ytemplate,memberpreds,ensemble