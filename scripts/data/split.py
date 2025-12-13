#!/usr/bin/env python

import os
import json
import glob
import h5py
import logging
import warnings
import numpy as np
import xarray as xr
from scripts.utils import Config

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

config = Config()
FILEDIR    = config.interimdir
SAVEDIR    = config.splitsdir
TRAINRANGE = config.trainrange
VALIDRANGE = config.validrange
TESTRANGE  = config.testrange

def split(splitrange,filedir=FILEDIR):
    '''
    Purpose: Load all NetCDF files into a single xr.Dataset for a given split.
    Args:
    - splitrange (tuple[int,int]): inclusive year range for the split
    - filedir (str): directory containing the NetCDF files (defaults to FILEDIR)
    Returns:
    - xr.Dataset: split Dataset
    '''
    filepaths = sorted(glob.glob(os.path.join(filedir,'*.nc')))
    datavars = {}
    for filepath in filepaths:
        da   = xr.open_dataarray(filepath,engine='h5netcdf')
        dims = tuple(dim for dim in ('lat','lon','lev','time') if dim in da.dims)
        da   = da.transpose(*dims) if dims else da
        datavars[da.name] = da
    ds = xr.Dataset(datavars)
    ds = ds.sel(time=(ds.time.dt.year>=splitrange[0])&(ds.time.dt.year<=splitrange[1]))    
    return ds

def stats(trainds,savedir=SAVEDIR):
    '''
    Purpose: Compute training-set statistics for each variable and save them to JSON. Statistics are not calculated 
    for land fraction and quadrature weights, and precipitation is log1p-transformed before statistics are calculated.
    Args:
    - trainds (xr.Dataset): training Dataset
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - dict[str,float]: the training set mean/standard deviation for select variables in 'trainds'
    '''
    stats = {}
    for varname,da in trainds.data_vars.items():
        if varname in ('lf','darea','dlev','dtime'):
            continue
        elif varname=='pr':
            arr = np.log1p(da.values)
        else:
            arr = da.values
        stats[f'{varname}_mean'] = float(np.nanmean(arr.ravel()))
        stats[f'{varname}_std']  = float(np.nanstd(arr.ravel()))
    filename = 'stats.json'
    os.makedirs(savedir,exist_ok=True)
    filepath = os.path.join(savedir,filename)
    with open(filepath,'w',encoding='utf-8') as f:
        json.dump(stats,f)
    logger.info(f'   Wrote statistics to {filename}')
    return stats

def normalize(ds,stats):
    '''
    Purpose: Normalize an xr.Dataset using training statistics (for all variables except land fraction and quadrature weights). For 
    precipitation, we use a log1p-transform before normalization. 
    Args:
    - ds (xr.Dataset): Dataset to normalize
    - stats (dict): normalization mean/standard deviation computed on the training set
    Returns:
    - xr.Dataset: normalized Dataset
    '''
    datavars = {}
    for varname,da in ds.data_vars.items():
        if varname in ('lf','darea','dlev','dtime'):
            datavars[varname] = da
            continue
        mean = stats[f'{varname}_mean']
        std  = stats[f'{varname}_std']
        if varname=='pr':
            norm   = (np.log1p(da.values)-mean)/std
            suffix = ' (log1p-transformed and standardized)'
        else:
            norm   = (da.values-mean)/std
            suffix = ' (standardized)'
        normda = xr.DataArray(norm.astype(np.float32),dims=da.dims,coords=da.coords,name=da.name)
        normda.attrs = dict(long_name=da.attrs.get('long_name',da.name)+suffix,units='N/A')
        datavars[varname] = normda
    normds = xr.Dataset(datavars,coords=ds.coords)
    return normds

def save(ds,splitname,timechunksize=2208,savedir=SAVEDIR):
    '''
    Purpose: Save an xr.Dataset to an HDF5 file, then verify the write by reopening.
    Args:
    - ds (xr.Dataset): Dataset to save
    - splitname (str): 'train' | 'valid' | 'test'
    - timechunksize (int): chunk size for the 'time' dimension (defaults to 2,208 for 3-month chunks)
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - bool: True if write and verification succeed, False otherwise
    '''
    os.makedirs(savedir,exist_ok=True)
    filename = f'{splitname}.h5'
    filepath = os.path.join(savedir,filename)
    logger.info(f'   Attempting to save {filename}...')
    ds.load()
    encoding = {}
    for varname,da in ds.data_vars.items():
        chunks = []
        for dim,size in zip(da.dims,da.shape):
            if dim=='time':
                chunks.append(min(timechunksize,size))
            else:
                chunks.append(size)
        encoding[varname] = {'chunksizes':tuple(chunks),'dtype':da.dtype}
    try:
        ds.to_netcdf(filepath,engine='h5netcdf',encoding=encoding)
        with h5py.File(filepath,'r'):
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

if __name__=='__main__':
    logger.info('Setting up splits...')
    splits = [
        ('train',TRAINRANGE),
        ('valid',VALIDRANGE),
        ('test',TESTRANGE)]
    logger.info('Creating and saving normalized data splits...')
    for splitname,splitrange in splits:
        splitds = split(splitrange)
        if splitname=='train':
            trainstats = stats(splitds)
        ds = normalize(splitds,trainstats)
        save(ds,splitname)
        del ds