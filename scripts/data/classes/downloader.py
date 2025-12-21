#!/usr/bin/env python

import os
import re
import sys
import fsspec
import logging
import numpy as np
import xarray as xr
import planetary_computer
from datetime import datetime
import pystac_client as pystac

logger = logging.getLogger(__name__)

class DataDownloader:

    def __init__(self,author,email,savedir,latrange,lonrange,levrange,years,months):
        '''
        Purpose: Initialize DataDownloader with configuration parameters.
        Args:
        - author (str): author name
        - email (str): author email
        - savedir (str): directory to save output files
        - latrange (tuple[float,float]): latitude range
        - lonrange (tuple[float,float]): longitude range
        - levrange (tuple[float,float]): pressure level range
        - years (list[int]): years to include
        - months (list[int]): months to include
        '''
        self.author   = author
        self.email    = email
        self.savedir  = savedir
        self.latrange = latrange
        self.lonrange = lonrange
        self.levrange = levrange
        self.years    = years
        self.months   = months

    def retrieve_era5(self):
        '''
        Purpose: Retrieve the ERA5 (ARCO) Zarr store from Google Cloud.
        Returns:
        - xr.Dataset: ERA5 Dataset on success, exits on failure
        '''
        try:
            store = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
            ds = xr.open_zarr(store,storage_options=dict(token='anon'))
            logger.info('   Successfully retrieved ERA5')
            return ds
        except Exception:
            logger.exception('   Failed to retrieve ERA5')
            sys.exit(1)

    def retrieve_imerg(self):
        '''
        Purpose: Retrieve the GPM IMERG V06 Zarr store from Microsoft Planetary Computer.
        Returns:
        - xr.Dataset: IMERG V06 Dataset on success, exits on failure
        '''
        try:
            store   = 'https://planetarycomputer.microsoft.com/api/stac/v1'
            catalog = pystac.Client.open(store,modifier=planetary_computer.sign_inplace)
            assets  = catalog.get_collection('gpm-imerg-hhr').assets['zarr-abfs']
            mapper  = fsspec.get_mapper(assets.href,**assets.extra_fields['xarray:storage_options'])
            ds      = xr.open_zarr(mapper,chunks={},consolidated=True)
            logger.info('   Successfully retrieved IMERG')
            return ds
        except Exception:
            logger.exception('   Failed to retrieve IMERG')
            sys.exit(1)

    def standardize(self,da):
        '''
        Purpose: Standardize dimension names, data types, and order of an xr.DataArray.
        Args:
        - da (xr.DataArray): input DataArray
        Returns:
        - xr.DataArray: standardized DataArray
        '''
        dimnames   = {'latitude':'lat','longitude':'lon','level':'lev'}
        da         = da.rename({old:new for old,new in dimnames.items() if old in da.dims})
        targetdims = [dim for dim in ('lat','lon','lev','time') if dim in da.dims]
        extradims  = [dim for dim in da.dims if dim not in targetdims]
        da = da.drop_dims(extradims) if extradims else da
        for dim in da.dims:
            if dim=='time':
                if da.coords[dim].dtype.kind!='M':
                    da.coords[dim] = da.indexes[dim].to_datetimeindex()
                da = da.sel(time=~da.time.to_index().duplicated(keep='first'))
            else:
                da.coords[dim] = da.coords[dim].astype('float32')
                if dim=='lon':
                    da.coords[dim] = (da.coords[dim]+180.0)%360.0-180.0
        da = da.sortby(targetdims).transpose(*targetdims)
        return da

    def subset(self,da,radius=0):
        '''
        Purpose: Subset an xr.DataArray by horizontal domain, pressure levels, and time.
        Args:
        - da (xr.DataArray): input DataArray
        - radius (int): grid cells beyond domain bounds for regridding (defaults to 0)
        Returns:
        - xr.DataArray: subsetted DataArray
        '''
        if 'time' in da.dims:
            da = da.sel(time=(da.time.dt.year.isin(self.years))&(da.time.dt.month.isin(self.months)))
        if 'lev' in da.dims:
            levmin,levmax = self.levrange[0],self.levrange[1]
            da = da.sel(lev=slice(levmin,levmax))
        if radius:
            latpad = radius*float(np.abs(np.median(np.diff(da.lat.values))))
            lonpad = radius*float(np.abs(np.median(np.diff(da.lon.values))))
            latmin = max(float(da.lat.min()),self.latrange[0]-latpad)
            latmax = min(float(da.lat.max()),self.latrange[1]+latpad)
            lonmin = max(float(da.lon.min()),self.lonrange[0]-lonpad)
            lonmax = min(float(da.lon.max()),self.lonrange[1]+lonpad)
        else:
            latmin,latmax = self.latrange[0],self.latrange[1]
            lonmin,lonmax = self.lonrange[0],self.lonrange[1]
        da = da.sel(lat=slice(latmin,latmax),lon=slice(lonmin,lonmax))
        return da

    def create_dataset(self,da,shortname,longname,units):
        '''
        Purpose: Wrap an xr.DataArray into a xr.Dataset with metadata.
        Args:
        - da (xr.DataArray): input DataArray
        - shortname (str): variable name
        - longname (str): variable description
        - units (str): variable units
        Returns:
        - xr.Dataset: Dataset containing the variable and metadata
        '''
        ds = da.to_dataset(name=shortname)
        ds[shortname].attrs = dict(long_name=longname,units=units)
        if 'lat' in ds.coords:
            ds.lat.attrs  = dict(long_name='Latitude',units='°N')
        if 'lon' in ds.coords:
            ds.lon.attrs  = dict(long_name='Longitude',units='°E')
        if 'lev' in ds.coords:
            ds.lev.attrs  = dict(long_name='Pressure level',units='hPa')
        if 'time' in ds.coords:
            ds.time.attrs = dict(long_name='Time')
        ds.attrs = dict(history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {self.author} ({self.email})')
        logger.info(f'   {longname} size: {ds.nbytes*1e-9:.3f} GB')
        return ds

    def process(self,da,shortname,longname,units,radius=0):
        '''
        Purpose: Apply standardize(), subset(), and create_dataset() in sequence.
        Args:
        - da (xr.DataArray): input DataArray
        - shortname (str): variable name
        - longname (str): variable description
        - units (str): variable units
        - radius (int): grid cells beyond domain bounds (defaults to 0)
        Returns:
        - xr.Dataset: processed Dataset
        '''
        da = self.standardize(da)
        da = self.subset(da,radius)
        ds = self.create_dataset(da,shortname,longname,units)
        return ds

    def save(self,ds,timechunksize=2208):
        '''
        Purpose: Save a Dataset to NetCDF and verify by reopening.
        Args:
        - ds (xr.Dataset): Dataset to save
        - timechunksize (int): chunk size for time dimension (defaults to 2,208 for 3-month chunks)
        Returns:
        - bool: True if save successful, False otherwise
        '''
        os.makedirs(self.savedir,exist_ok=True)
        shortname = list(ds.data_vars)[0]
        longname  = ds[shortname].attrs['long_name']
        filename  = re.sub(r'\s+','_',longname)+'.nc'
        filepath  = os.path.join(self.savedir,filename)
        logger.info(f'   Attempting to save {filename}...')
        ds.load()
        ds[shortname].encoding = {}
        chunks = []
        for dim,size in zip(ds[shortname].dims,ds[shortname].shape):
            chunks.append(min(timechunksize,size) if dim=='time' else size)
        encoding = {shortname:{'chunksizes':tuple(chunks)}}
        try:
            ds.to_netcdf(filepath,engine='h5netcdf',encoding=encoding)
            xr.open_dataset(filepath,engine='h5netcdf').close()
            logger.info('      File write successful')
            return True
        except Exception:
            logger.exception('      Failed to save or verify')
            return False