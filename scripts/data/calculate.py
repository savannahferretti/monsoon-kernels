#!/usr/bin/env python

import os
import xesmf
import logging
import warnings
import numpy as np
import xarray as xr
from utils import Config
from datetime import datetime

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

config = Config()
AUTHOR   = config.author
EMAIL    = config.email
FILEDIR  = config.rawdir
SAVEDIR  = config.interimdir
LATRANGE = config.latrange
LONRANGE = config.lonrange

def retrieve(longname,filedir=FILEDIR):
    '''
    Purpose: Lazily import in a NetCDF file as an xr.DataArray and, if applicable, ensure pressure levels 
    are ascending (e.g., [500,550,600,...] hPa).
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
    p = refda.lev.expand_dims({'lat':refda.lat,'lon':refda.lon,'time':refda.time}).transpose('lat','lon','lev','time')
    return p

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
    Purpose: Regrids a DataArray to a 1.0° x 1.0° target grid that extends 1 cell beyond 'latrange'/'lonrange' in each direction.
    Args:
    - da (xr.DataArray): input DataArray (with halo)
    - latrange (tuple[float,float]): target latitude range (defaults to LATRANGE)
    - lonrange (tuple[float,float]): target longitude range (defaults to LONRANGE)
    Returns:
    - xr.DataArray: DataArray regridded to target domain
    '''
    targetlats = np.arange(latrange[0]-1.0,latrange[1]+2.0,1.0)
    targetlons = np.arange(lonrange[0]-1.0,lonrange[1]+2.0,1.0)
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
    Purpose: Calculate saturation specific humidity (qₛ) using Eq. 4 from Miller SFK. (2018), Atmos. 
    Humidity Eq. Plymouth State Wea. Ctr.
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
    - xr.DataArray: RH DataArray (%)
    '''
    qs = calc_qs(p,t)         
    rh = (q/qs)*100.0      
    rh = rh.clip(min=0.0,max=100.0)
    return rh

def calc_thetae(p,t,q=None):
    '''
    Purpose: Calculate (unsaturated or saturated) equivalent potential temperature (θₑ) using Eqs. 43 and 55 
    from Bolton D. (1980), Mon. Wea. Rev.       
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

def calc_quadrature_weights(refda,rearth=6.371e6):
    '''
    Purpose: Compute ΔA Δp Δt quadrature weights for a 4D grid (lat, lon, lev, time). These weights ensure that 
    a sum over grid points approximates a physical integral over space and time.
    Args:
    - refda (xr.DataArray): reference DataArray containing 'lat', 'lon', 'lev', and 'time' coordinates
    - rearth (float): Earth's radius in meters (defaults to 6,371,000)
    Returns:
    - xr.DataArray: quadrature weights with dims ('lat', 'lon', 'lev', 'time')
    '''
    refda = refda.transpose('lat','lon','lev','time')
    lats  = refda.lat.values
    lons  = refda.lon.values
    levs  = refda.lev.values
    times = np.arange(refda.time.size,dtype=np.float32)
    def spacing(coord):
        '''
        Purpose: Estimate spacing between neighboring coordinate points using centered differences in the interior 
        and one-sided differences at the boundaries.
        Args:
        - coord (np.ndarray): 1D array of monotonically increasing coordinate values
        Returns:
        - np.ndarray: 1D array of estimated spacing between grid points with the same length as 'coord'
        '''
        spacing = np.empty_like(coord)
        spacing[1:-1] = 0.5*(coord[2:]-coord[:-2])
        spacing[0]    = coord[1]-coord[0]
        spacing[-1]   = coord[-1]-coord[-2]
        return np.abs(spacing)
    dlat  = spacing(np.deg2rad(lats))
    dlon  = spacing(np.deg2rad(lons))
    dlev  = spacing(levs)
    dtime = spacing(times)
    area  = ((rearth**2)*np.cos(np.deg2rad(lats))*dlat).reshape(lats.size,1)*dlon.reshape(1,lons.size)
    weights = area.reshape(lats.size,lons.size,1,1)*dlev.reshape(1,1,levs.size,1)*dtime.reshape(1,1,1,times.size).astype(np.float32)
    weightsda = xr.DataArray(weights,dims=('lat','lon','lev','time'),coords=dict(lat=refda.lat,lon=refda.lon,lev=refda.lev,time=refda.time))
    return weightsda

def dataset(da,shortname,longname,units,author=AUTHOR,email=EMAIL):
    '''
    Purpose: Wrap a standardized xr.DataArray into an xr.Dataset, preserving coordinates and setting 
    variable and global metadata.
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
    dims = [dim for dim in ('lat','lon','lev','time') if dim in da.dims]
    da = da.transpose(*dims)
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
    logger.info(f'   {shortname}: {ds.nbytes*1e-9:.3f} GB')
    return ds

def save(ds,timechunksize=2208,savedir=SAVEDIR):
    '''
    Purpose: Save an xr.Dataset to a NetCDF file in the specified directory, then verify the write by reopening.
    Args:
    - ds (xr.Dataset): Dataset to save
    - timechunksize (int): chunk size for the 'time' dimension (defaults to 2,208 for 3-month chunks)
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - bool: True if write and verification succeed, otherwise False
    '''
    os.makedirs(savedir,exist_ok=True)
    shortname = list(ds.data_vars)[0]
    filename  = f'{shortname}.nc'
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
    logger.info('Importing all raw variables...')
    ps  = retrieve('ERA5_surface_pressure')
    t   = retrieve('ERA5_air_temperature')
    q   = retrieve('ERA5_specific_humidity')
    lf  = retrieve('ERA5_land_fraction')
    lhf = retrieve('ERA5_mean_surface_latent_heat_flux')
    shf = retrieve('ERA5_mean_surface_sensible_heat_flux')
    pr  = retrieve('IMERG_V06_precipitation_rate')
    logger.info('Resampling/regridding variables...')
    ps  = regrid(ps).load()
    t   = regrid(t).load()
    q   = regrid(q).load()
    lf  = regrid(lf).load()
    lhf = regrid(lhf).load()
    shf = regrid(shf).load()
    pr  = regrid(resample(pr)).clip(min=0).load()
    logger.info('Calculating relative humidity and equivalent potential temperature terms...')
    p          = create_p_array(q)
    rh         = calc_rh(p,t,q)
    thetae     = calc_thetae(p,t,q)
    thetaestar = calc_thetae(p,t)
    logger.info('Calculating quadrature weights...')
    quadweights = calc_quadrature_weights(t)
    logger.info('Creating datasets...')
    dslist = [        
        dataset(t,'t','Air temperature','K'),
        dataset(q,'q','Specific humidity','kg/kg'),
        dataset(rh,'rh','Relative humidity','%'),
        dataset(thetae,'thetae','Equivalent potential temperature','K'),
        dataset(thetaestar,'thetaestar','Saturated equivalent potential temperature','K'),
        dataset(lf,'lf','Land fraction','0-1'),
        dataset(lhf,'lhf','Mean surface latent heat flux','W/m²'),
        dataset(shf,'shf','Mean surface sensible heat flux','W/m²'),
        dataset(pr,'pr','Precipitation rate','mm/hr'),
        dataset(quadweights,'quadweights','Quadrature integration weights','m² hPa hr')]
    logger.info('Saving datasets...')
    for ds in dslist:
        save(ds)
        del ds