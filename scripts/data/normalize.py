#!/usr/bin/env python

import os
import json
import logging
import warnings
import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR     = '/global/cfs/cdirs/m4334/sferrett/monsoon-kernels/data/splits'
TARGETVAR   = 'pr'
CHUNKSIZE   = 2208   

def load(splitname,filedir=FILEDIR):
    '''
    Purpose: Load a data split as an xr.Dataset.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - filedir (str): directory containing split files (defaults to FILEDIR)
    Returns:
    - xr.Dataset: Dataset for the requested split
    '''
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    splitds  = xr.open_dataset(filepath,engine='h5netcdf')
    return splitds.load()
    
def reshape(da):
    '''
    Purpose: Convert an xr.DataArray into a 2D NumPy array suitable for NN I/O.
    Args:
    - da (xr.DataArray): 3D or 4D DataArray
    Returns:
    - np.ndarray: shape (nsamples, nfeatures); for 3D, nfeatures=1, for 4D, nfeatures equals the size of the 'lev' dimension
    '''
    if 'lev' in da.dims:
        arr = da.transpose('time','lat','lon','lev').values.reshape(-1,da.lev.size)
    else:
        arr = da.transpose('time','lat','lon').values.reshape(-1,1)
    return arr

def calc_save_stats(trainds,targetvar=TARGETVAR,filedir=FILEDIR):
    '''
    Purpose: Compute training-set statistics for each variable and save to JSON. For 4D input variables 
    (with 'lev'), compute statistics using elements where levmask==1. For the target variable, compute statistics on log1p(target).
    Args:
    - trainds(xr.Dataset): training Dataset
    - targetvar (str): target variable name (defaults to TARGETVAR)
    - filedir (str): output directory for saving the JSON file (defaults to FILEDIR)
    Returns:
    - dict[str,float]: the training set mean/std for each variable in 'trainds'
    '''
    stats = {}
    levmask = reshape(trainds['levmask']).astype(bool)
    for varname in trainds.data_vars:
        if varname=='levmask':
            continue
        da  = trainds[varname]
        arr = reshape(da)
        if varname==targetvar:
            flat = np.log1p(arr).ravel()
        else:
            flat = arr[levmask] if ('lev' in da.dims) else arr.ravel()
        stats[f'{varname}_mean'] = float(np.nanmean(flat))
        stats[f'{varname}_std']  = float(np.nanstd(flat))
    filename = 'stats.json'
    filepath = os.path.join(filedir,filename)
    with open(filepath,'w',encoding='utf-8') as f:
        json.dump(stats,f)
    logger.info(f'   Wrote statistics to {filename}')
    return stats

def normalize(da,stats,levmask,targetvar=TARGETVAR):
    '''
    Purpose: Z-score normalize an xr.DataArray using provided statistics and an optional mask. For 4D inputs, multiply by the 
    level mask to gate invalid pressure levels to 0. For the target variable, we log1p-transform then normalize.  
    Args:
    - da (xr.DataArray): variable DataArray to normalize
    - stats (dict): normalization mean/std
    - levmask (np.ndarray): below-surface level mask
    - targetvar (str): target variable name (defaults to TARGETVAR)
    Returns:
    - xr.DataArray: normalized DataArray
    '''
    arr = reshape(da)
    if da.name==targetvar:
        arr  = np.log1p(arr)
        norm = (arr-stats[f'{da.name}_mean'])/stats[f'{da.name}_std']
    else:
        norm = (arr-stats[f'{da.name}_mean'])/stats[f'{da.name}_std']
        if 'lev' in da.dims:
            norm = norm*levmask
    normda   = xr.DataArray(norm.astype(np.float32).reshape(da.shape),dims=da.dims,coords=da.coords,name=da.name)
    longname = da.attrs.get('long_name',da.name)
    if da.name==targetvar:
        normda.attrs = dict(long_name=f'{longname} (log1p-transform and Z-score normalization)',units='N/A')
    else:
        normda.attrs = dict(long_name=f'{longname} (Z-score normalization)',units='N/A')
    return normda

def process(splitname,stats,chunksize=CHUNKSIZE,filedir=FILEDIR):
    '''
    Purpose: Normalize an existing data split using 'stats' and save as an HDF5 file.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - stats (dict): normalization mean/std
    - chunksize (int): number of time steps to include for chunking (defaults to 2,208 for 3-month chunks)
    - filedir (str): directory containing split files/output directory (defaults to FILEDIR)
    Returns:
    - bool: True if writing and verification succeed, otherwise False
    '''
    logger.info(f'   Normalizing {splitname}')
    splitds  = load(splitname)
    levmask  = reshape(splitds['levmask']).astype(bool)
    datavars = {} 
    for varname in splitds.data_vars:
        if varname=='levmask':
            datavars['levmask'] = splitds['levmask'].astype('uint8')
        else:
            datavars[varname] = normalize(splitds[varname],stats,levmask)
    ds = xr.Dataset(datavars)
    encoding = {}
    for varname,da in ds.data_vars.items():
        chunks = (chunksize,da.lat.size,da.lon.size,da.lev.size) if 'lev' in da.dims else (chunksize,da.lat.size,da.lon.size)
        encoding[varname] = {
            'compression':'lzf',
            'shuffle':True,
            'chunksizes':chunks,
            'dtype':('uint8' if varname=='levmask' else da.dtype)}
    filename = f'norm{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    logger.info(f'      Attempting to save {filename} ...')
    try:
        ds.to_netcdf(filepath,engine='h5netcdf',encoding=encoding)
        with xr.open_dataset(filepath,engine='h5netcdf') as _:
            pass
        logger.info('         File write successful')
        return True
    except Exception:
        logger.exception('         Failed to save or verify')
        return False

if __name__=='__main__':
    logger.info('Loading the training set...')
    trainds = load('train')
    logger.info('Computing training set normalization statistics...')
    stats = calc_save_stats(trainds)
    del trainds
    logger.info('Creating normalized splits and saving...')
    for splitname in ('train','valid','test'):
        process(splitname,stats)
    del stats