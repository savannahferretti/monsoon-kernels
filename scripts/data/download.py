#!/usr/bin/env python

import os
import re
import sys
import fsspec
import logging
import warnings
import numpy as np
import xarray as xr
import planetary_computer
from datetime import datetime
import pystac_client as pystac

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Config

logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

config = Config()
AUTHOR   = config.author
EMAIL    = config.email
SAVEDIR  = config.rawdir
LATRANGE = config.latrange
LONRANGE = config.lonrange
LEVRANGE = config.levrange
YEARS    = config.years
MONTHS   = config.months

# class DataDownloader:
#     def __init__(self, config):
#         self.config = config
    
#     def download_era5(self): ...
#     def download_imerg(self): ...
#     def download_all(self): ...

def retrieve_era5():
    '''
    Purpose: Retrieve the ERA5 (ARCO) Zarr store from Google Cloud and return it as an xr.Dataset.
    Returns:
    - xr.Dataset: ERA5 Dataset on success, the program exits if access fails
    '''
    try:
        store = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
        ds = xr.open_zarr(store,storage_options=dict(token='anon'))  
        logger.info('   Successfully retrieved ERA5')
        return ds
    except Exception:
        logger.exception('   Failed to retrieve ERA5')
        sys.exit(1)

def retrieve_imerg():
    '''
    Purpose: Retrieve the GPM IMERG V06 Zarr store from Microsoft Planetary Computer and return it as an xr.Dataset.
    Returns: 
    - xr.Dataset: IMERG V06 Dataset on success, the program exits if access fails
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
    
def standardize(da):
    '''
    Purpose: Standardize the dimension names, data types, and order of an xr.DataArray.
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

def subset(da,radius=0,latrange=LATRANGE,lonrange=LONRANGE,levrange=LEVRANGE,years=YEARS,months=MONTHS):
    '''
    Purpose: Subset an xr.DataArray by a horizontal domain, and, if present, by pressure levels and time.
    Args:
    - da (xr.DataArray): input DataArray
    - radius (int): number of grid cells to include beyond domain bounds for regridding (defaults to 0, 
      which disables the radius)
    - latrange (tuple[float,float]): latitude range (defaults to LATRANGE)
    - lonrange (tuple[float,float]): longitude range (defaults to LONRANGE)
    - levrange (tuple[float,float]): pressure level range in hPa (defaults to LEVRANGE)
    - years (list[int]): years to include (defaults to YEARS)
    - months (list[int]): months to include (defaults to MONTHS)
    Returns:
    - xr.DataArray: subsetted DataArray
    ''' 
    if 'time' in da.dims:
        da = da.sel(time=(da.time.dt.year.isin(years))&(da.time.dt.month.isin(months)))
    if 'lev' in da.dims:
        levmin,levmax = levrange[0],levrange[1]
        da = da.sel(lev=slice(levmin,levmax)) 
    if radius:
        latpad = radius*float(np.abs(np.median(np.diff(da.lat.values))))
        lonpad = radius*float(np.abs(np.median(np.diff(da.lon.values))))
        latmin = max(float(da.lat.min()),latrange[0]-latpad)
        latmax = min(float(da.lat.max()),latrange[1]+latpad)
        lonmin = max(float(da.lon.min()),lonrange[0]-lonpad)
        lonmax = min(float(da.lon.max()),lonrange[1]+lonpad)
    else:
        latmin,latmax = latrange[0],latrange[1]
        lonmin,lonmax = lonrange[0],lonrange[1]
    da = da.sel(lat=slice(latmin,latmax),lon=slice(lonmin,lonmax))   
    return da

def dataset(da,shortname,longname,units,author=AUTHOR,email=EMAIL):
    '''
    Purpose: Wrap a standardized xr.DataArray into an xr.Dataset, preserving coordinates and setting 
    variable and global metadata.
    Args:
    - da (xr.DataArray): input DataArray
    - shortname (str): variable name
    - longname (str): variable description
    - units (str): variable units
    - author (str): author name (defaults to AUTHOR)
    - email (str): author email (defaults to EMAIL)    
    Returns:
    - xr.Dataset: Dataset containing the variable named 'shortname' and metadata
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
    ds.attrs = dict(history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {author} ({email})')
    logger.info(f'   {longname} size: {ds.nbytes*1e-9:.3f} GB')
    return ds

def process(da,shortname,longname,units,radius=0,latrange=LATRANGE,lonrange=LONRANGE,levrange=LEVRANGE,
            years=YEARS,months=MONTHS,author=AUTHOR,email=EMAIL):
    '''
    Purpose: Convert an xr.DataArray into an xr.Dataset by applying the standardize(), subset(), and 
    dataset() functions in sequence.
    Args:
    - da (xr.DataArray): input DataArray
    - shortname (str): variable name
    - longname (str): variable description 
    - units (str): variable units
    - radius (int): number of grid cells to include beyond domain bounds for regridding (defaults to 0, 
      which disables the radius)
    - latrange (tuple[float,float]): latitude range (defaults to LATRANGE)
    - lonrange (tuple[float,float]): longitude range (defaults to LONRANGE)
    - levrange (tuple[float,float]): pressure level range in hPa (defaults to LEVRANGE)
    - years (list[int]): years to include (defaults to YEARS)
    - months (list[int]): months to include (defaults to MONTHS)
    - author (str): author name (defaults to AUTHOR)
    - email (str): author email (defaults to EMAIL)
    Returns:
    - xr.Dataset: processed Dataset
    ''' 
    da = standardize(da)
    da = subset(da,radius,latrange,lonrange,levrange,years,months)
    ds = dataset(da,shortname,longname,units,author,email)
    return ds

def save(ds,timechunksize=2208,savedir=SAVEDIR):
    '''
    Purpose: Save an xr.Dataset to a NetCDF file, then verify by reopening.
    Args:
    - ds (xr.Dataset): Dataset to save
    - timechunksize (int): chunk size for the 'time' dimension (defaults to 2,208 for 3-month chunks)
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - bool: True if write and verification succeed, False otherwise
    '''
    os.makedirs(savedir,exist_ok=True)
    shortname = list(ds.data_vars)[0]
    longname  = ds[shortname].attrs['long_name']
    filename  = re.sub(r'\s+','_',longname)+'.nc'
    filepath  = os.path.join(savedir,filename)
    logger.info(f'   Attempting to save {filename}...') 
    ds.load()
    ds[shortname].encoding = {}
    encodingchunks = []
    for dim,size in zip(ds[shortname].dims,ds[shortname].shape):
        if dim=='time':
            encodingchunks.append(min(timechunksize,size))
        else:
            encodingchunks.append(size)
    encoding = {shortname:{'chunksizes':tuple(encodingchunks)}}
    try:
        ds.to_netcdf(filepath,engine='h5netcdf',encoding=encoding)
        xr.open_dataset(filepath,engine='h5netcdf').close()
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

if __name__=='__main__':
    logger.info('Retrieving ERA5 and IMERG data...')
    era5  = retrieve_era5()
    imerg = retrieve_imerg()
    logger.info('Extracting variable data...')
    psdata  = era5.surface_pressure/100.0
    tdata   = era5.temperature
    qdata   = era5.specific_humidity
    lfdata  = era5.land_sea_mask
    lhfdata = era5.mean_surface_latent_heat_flux
    shfdata = era5.mean_surface_sensible_heat_flux
    prdata  = imerg.precipitationCal
    del era5,imerg
    logger.info('Creating datasets...')
    dslist = [
        process(psdata,'ps','ERA5 surface pressure','hPa',radius=4),
        process(tdata,'t','ERA5 air temperature','K',radius=4),
        process(qdata,'q','ERA5 specific humidity','kg/kg',radius=4),
        process(lfdata,'lf','ERA5 land fraction','0-1',radius=4),
        process(lhfdata,'lhf','ERA5 mean surface latent heat flux','W/m²',radius=4),
        process(shfdata,'shf','ERA5 mean surface sensible heat flux','W/m²',radius=4),
        process(prdata,'pr','IMERG V06 precipitation rate','mm/hr',radius=10)]
    del psdata,tdata,qdata,lfdata,lhfdata,shfdata,prdata
    logger.info('Saving datasets...')
    for ds in dslist:
        save(ds)
        del ds