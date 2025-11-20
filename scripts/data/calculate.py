#!/usr/bin/env python

import os
import xesmf
import logging
import warnings
import numpy as np
import xarray as xr
from datetime import datetime

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

AUTHOR   = 'Savannah L. Ferretti'      
EMAIL    = 'savannah.ferretti@uci.edu' 
FILEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-kernels/data/raw'
SAVEDIR  = '/global/cfs/cdirs/m4334/sferrett/monsoon-kernels/data/interim'
LATRANGE = (5.0,25.0) 
LONRANGE = (60.0,90.0)

def retrieve(longname,filedir=FILEDIR):
    '''
    Purpose: Lazily import in a NetCDF file as an xr.DataArray and, if applicable, ensure pressure levels are ascending (e.g., [500,550,600,...] hPa).
    Args:
    - longname (str): variable long name/description
    - filedir (str): directory containing the file (defaults to FILEDIR)
    Returns:
    - xr.DataArray: loaded DataArray with levels ordered (if applicable) 
    '''
    filename = f'{longname}.nc'
    filepath = os.path.join(filedir,filename)
    da = xr.open_dataarray(filepath,engine='h5netcdf')
    if 'lev' in da.dims:
        if not np.all(np.diff(da['lev'].values)>0):
            da = da.sortby('lev')
            logger.info(f'   Levels for {filename} were reordered to ascending')
    return da
    
def create_p_array(refda):
    '''
    Purpose: Create a pressure xr.DataArray from the 'lev' dimension.
    Args:
    - refda (xr.DataArray): reference DataArray containing 'lev'
    Returns:
    - xr.DataArray: pressure DataArray
    '''
    p = refda.lev.expand_dims({'time':refda.time,'lat':refda.lat,'lon':refda.lon}).transpose('lev','time','lat','lon')
    return p

def create_level_mask(refda,ps):
    '''
    Purpose: Create a below-surface level mask; 1 where levels exist (lev ≤ ps), else 0.
    - refda (xr.DataArray): reference DataArray containing 'lev'
    - ps (xr.DataArray): surface pressure (hPa)
    Returns:
    - xr.DataArray: DataArray of 0's (invalid levels) or 1's (valid levels)
    '''
    levmask = (refda.lev<=ps).transpose('time','lat','lon','lev').astype('uint8')
    return levmask

def resample(da):
    '''
    Purpose: Compute a centered hourly mean (uses the two half-hour samples that straddle each hour; falls back to 
    one at boundaries).
    Args:
    - da (xr.DataArray): input DataArray
    Returns:
    - xr.DataArray: DataArray resampled at on-the-hour timestamps
    '''
    da = da.rolling(time=2,center=True,min_periods=1).mean()
    da = da.sel(time=da.time.dt.minute==0)
    return da
    
def regrid(da,latrange=LATRANGE,lonrange=LONRANGE):
    '''
    Purpose: Regrids a DataArray to a 1° x 1° target grid.
    Args:
    - da (xr.DataArray): input DataArray (with halo)
    - latrange (tuple[float,float]): target latitude range (defaults to LATRANGE)
    - lonrange (tuple[float,float]): target longitude range (defaults to LONRANGE)
    Returns:
    - xr.DataArray: DataArray regridded to target domain
    '''
    targetlats = np.arange(latrange[0],latrange[1]+1.0,1.0)
    targetlons = np.arange(lonrange[0],lonrange[1]+1.0,1.0)
    targetgrid = xr.Dataset({'lat':(['lat'],targetlats),'lon':(['lon'],targetlons)})
    regridder  = xesmf.Regridder(da,targetgrid,method='conservative')
    da = regridder(da,keep_attrs=True)
    return da
    
def calc_es(t):
    '''
    Purpose: Calculate saturation vapor pressure (eₛ) using Eqs. 17 and 18 from Huang J. (2018), J. Appl. Meteorol. Climatol.
    Args:
    - t (xr.DataArray): temperature DataArray (K) 
    Returns:
    - xr.DataArray: eₛ DataArray (hPa)
    '''    
    tc = t-273.15
    eswat = np.exp(34.494-(4924.99/(tc+237.1)))/((tc+105.0)**1.57)
    esice = np.exp(43.494-(6545.8/(tc+278.0)))/((tc+868.0)**2.0)
    es = xr.where(tc>0.0,eswat,esice)/100.0
    return es

def calc_qs(p,t):
    '''
    Purpose: Calculate saturation specific humidity (qₛ) using Eq. 4 from Miller SFK. (2018), Atmos. Humidity Eq. Plymouth State Wea. Ctr.
    Args:
    - p (xr.DataArray): pressure DataArray (hPa)
    - t (xr.DataArray): temperature DataArray (K)
    Returns:
    - xr.DataArray: qₛ DataArray (kg/kg)
    '''
    rv = 461.50   
    rd = 287.04   
    epsilon = rd/rv
    es = calc_es(t) 
    qs = (epsilon*es)/(p-es*(1.0-epsilon))
    return qs

def calc_rh(p,t,q):
    '''
    Purpose: Calculate relative humidity (RH) using saturation specific humidity (qₛ).
    Args:
    - p (xr.DataArray): pressure DataArray (hPa)
    - t (xr.DataArray): temperature DataArray (K)
    - q (xr.DataArray): specific humidity DataArray (kg/kg)
    Returns:
    - xr.DataArray: relative humidity DataArray (%)
    '''
    qs = calc_qs(p,t)         
    rh = (q/qs)*100.0      
    rh = rh.clip(min=0.0,max=100.0)
    return rh

def calc_thetae(p,t,q=None):
    '''
    Purpose: Calculate (unsaturated or saturated) equivalent potential temperature (θₑ) using Eqs. 43 and 55 from Bolton D. (1980), Mon. Wea. Rev.       
    Args:
    - p (xr.DataArray): pressure DataArray (hPa)
    - t (xr.DataArray): temperature DataArray (K)
    - q (xr.DataArray, optional): specific humidity DataArray (kg/kg); if None, saturated θₑ will be calculated
    Returns:
    - xr.DataArray: (unsaturated or saturated) θₑ DataArray (K)
    '''
    if q is None:
        q = calc_qs(p,t)
    p0 = 1000.0  
    rv = 461.5  
    rd = 287.04
    epsilon = rd/rv
    r  = q/(1.0-q) 
    e  = (p*r)/(epsilon+r)
    tl = 2840.0/(3.5*np.log(t)-np.log(e)-4.805)+55.0
    thetae = t*(p0/p)**(0.2854*(1.0-0.28*r))*np.exp((3.376/tl-0.00254)*1000.0*r*(1.0+0.81*r))
    return thetae

def dataset(da,shortname,longname,units,author=AUTHOR,email=EMAIL):
    '''
    Purpose: Wrap a standardized xr.DataArray into an xr.Dataset, preserving coordinates and setting variable and global metadata.
    Args:
    - da (xr.DataArray): input DataArray
    - shortname (str): variable name (abbreviation)
    - longname (str): variable long name/description
    - units (str): variable units
    - author (str): author name (defaults to AUTHOR)
    - email (str): author email (defaults to EMAIL)    
    Returns:
    - xr.Dataset: Dataset containing the variable named 'shortname' and metadata
    '''    
    dims = [dim for dim in ('time','lat','lon','lev') if dim in da.dims]
    da = da.transpose(*dims)
    ds = da.to_dataset(name=shortname)
    ds[shortname].attrs = dict(long_name=longname,units=units)
    if 'time' in ds.coords:
        ds.time.attrs = dict(long_name='Time')
    if 'lat' in ds.coords:
        ds.lat.attrs  = dict(long_name='Latitude',units='°N')
    if 'lon' in ds.coords:
        ds.lon.attrs  = dict(long_name='Longitude',units='°E')
    if 'lev' in ds.coords:
        ds.lev.attrs  = dict(long_name='Pressure level',units='hPa')
    ds.attrs = dict(history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {author} ({email})')
    logger.info(f'   {shortname}: {ds.nbytes*1e-9:.3f} GB')
    return ds
    
def save(ds,savedir=SAVEDIR):
    '''
    Purpose: Save an xr.Dataset to a NetCDF file in the specified directory, then verify the write by reopening.
    Args:
    - ds (xr.Dataset): Dataset to save
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - bool: True if write and verification succeed, otherwise False
    '''  
    os.makedirs(savedir,exist_ok=True)
    shortname = list(ds.data_vars)[0]
    filename  = f'{shortname}.nc' 
    filepath  = os.path.join(savedir,filename)
    encoding  = {name: {'dtype':('uint8' if name=='levmask' else 'float32')} for name in ds.data_vars}
    encoding.update({coord:{'dtype':'float32'} for coord in ('lat','lon','lev') if coord in ds.coords})
    logger.info(f'   Attempting to save {filename}...')   
    try:
        ds.to_netcdf(filepath,engine='h5netcdf',encoding=encoding)
        with xr.open_dataset(filepath,engine='h5netcdf') as _:
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

if __name__=='__main__':
    logger.info('Importing all raw variables...')
    pr  = retrieve('IMERG_V06_precipitation_rate')
    sdo = retrieve('ERA5_standard_deviation_of_orography')
    ps  = retrieve('ERA5_surface_pressure')
    t   = retrieve('ERA5_air_temperature')
    q   = retrieve('ERA5_specific_humidity')
    logger.info('Resampling/regridding variables...')
    pr  = regrid(resample(pr)).clip(min=0).load()
    sdo = regrid(sdo).load()
    ps  = regrid(ps).load()
    t   = regrid(t).load()
    q   = regrid(q).load()
    logger.info('Creating below-surface level mask...')
    levmask = create_level_mask(t,ps)
    logger.info('Calculating relative humidity and equivalent potential temperature terms...')
    p          = create_p_array(q)
    rh         = calc_rh(p,t,q)
    thetae     = calc_thetae(p,t,q)
    thetaestar = calc_thetae(p,t)
    thetaeplus = thetaestar-thetae
    logger.info('Creating datasets...')
    dslist = [
        dataset(pr,'pr','Precipitation rate','mm/hr'),
        dataset(sdo,'sdo','Standard deviation of orography','m'),
        dataset(t,'t','Air temperature','K'),
        dataset(q,'q','Specific humidity','kg/kg'),
        dataset(rh,'rh','Relative humidity','%'),
        dataset(thetae,'thetae','Equivalent potential temperature','K'),
        dataset(thetaestar,'thetaestar','Saturated equivalent potential temperature','K'),
        dataset(thetaeplus,'thetaeplus','Difference between saturated and unsaturated equivalent potential temperature','K'),
        dataset(levmask,'levmask','Below-surface level mask','N/A'),]
    logger.info('Saving datasets...')
    for ds in dslist:
        save(ds)
        del ds