#!/usr/bin/env python

import os
import json
import glob
import h5py
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

class DataSplitter:

    def __init__(self,filedir,savedir,trainrange,validrange,testrange):
        '''
        Purpose: Initialize DataSplitter with configuration parameters.
        Args:
        - filedir (str): directory containing NetCDF files
        - savedir (str): directory to save split files
        - trainrange (tuple[int,int]): inclusive year range for training split
        - validrange (tuple[int,int]): inclusive year range for validation split
        - testrange (tuple[int,int]): inclusive year range for test split
        '''
        self.filedir    = filedir
        self.savedir    = savedir
        self.trainrange = trainrange
        self.validrange = validrange
        self.testrange  = testrange

    def split(self,splitrange):
        '''
        Purpose: Load all NetCDF files into a single xr.Dataset for a given split.
        Args:
        - splitrange (tuple[int,int]): inclusive year range for the split
        Returns:
        - xr.Dataset: split Dataset
        '''
        filepaths = sorted(glob.glob(os.path.join(self.filedir,'*.nc')))
        datavars = {}
        for filepath in filepaths:
            da   = xr.open_dataarray(filepath,engine='h5netcdf')
            dims = tuple(dim for dim in ('lat','lon','lev','time') if dim in da.dims)
            da   = da.transpose(*dims) if dims else da
            datavars[da.name] = da
        ds = xr.Dataset(datavars)
        ds = ds.sel(time=(ds.time.dt.year>=splitrange[0])&(ds.time.dt.year<=splitrange[1]))
        return ds

    def calc_stats(self,trainds):
        '''
        Purpose: Compute training-set statistics for each variable and save to JSON.
        Args:
        - trainds (xr.Dataset): training Dataset
        Returns:
        - dict[str,float]: training set mean and standard deviation for select variables
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
        os.makedirs(self.savedir,exist_ok=True)
        filepath = os.path.join(self.savedir,filename)
        with open(filepath,'w',encoding='utf-8') as f:
            json.dump(stats,f)
        logger.info(f'   Wrote statistics to {filename}')
        return stats

    def normalize(self,ds,stats):
        '''
        Purpose: Normalize an xr.Dataset using training statistics.
        Args:
        - ds (xr.Dataset): Dataset to normalize
        - stats (dict): normalization mean and standard deviation from training set
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

    def save(self,ds,splitname,timechunksize=2208):
        '''
        Purpose: Save an xr.Dataset to an HDF5 file and verify by reopening.
        Args:
        - ds (xr.Dataset): Dataset to save
        - splitname (str): train, valid, or test
        - timechunksize (int): chunk size for time dimension (defaults to 2208)
        Returns:
        - bool: True if save successful, False otherwise
        '''
        os.makedirs(self.savedir,exist_ok=True)
        filename = f'{splitname}.h5'
        filepath = os.path.join(self.savedir,filename)
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
