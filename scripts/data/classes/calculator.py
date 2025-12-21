#!/usr/bin/env python

import os
import xesmf
import logging
import numpy as np
import xarray as xr
from datetime import datetime

logger = logging.getLogger(__name__)

class DataCalculator:

    def __init__(self,author,email,filedir,savedir,latrange,lonrange):
        '''
        Purpose: Initialize DataCalculator with configuration parameters.
        Args:
        - author (str): author name
        - email (str): author email
        - filedir (str): directory containing input NetCDF files
        - savedir (str): directory to save output files
        - latrange (tuple[float,float]): latitude range
        - lonrange (tuple[float,float]): longitude range
        '''
        self.author   = author
        self.email    = email
        self.filedir  = filedir
        self.savedir  = savedir
        self.latrange = latrange
        self.lonrange = lonrange

    def retrieve(self,longname):
        '''
        Purpose: Lazily import in a NetCDF file as an xr.DataArray with ascending pressure levels, if applicable 
        (e.g., [500,550,600,...] hPa).
        Args:
        - longname (str): variable description
        Returns:
        - xr.DataArray: DataArray with levels ordered (if applicable)
        '''
        filename = f'{longname}.nc'
        filepath = os.path.join(self.filedir,filename)
        da = xr.open_dataarray(filepath,engine='h5netcdf')
        if 'lev' in da.dims:
            if not np.all(np.diff(da.lev.values)>0):
                da = da.sortby('lev')
                logger.info(f'   Levels for {filename} were reordered to ascending')
        return da

    def create_p_array(self,refda):
        '''
        Purpose: Create a pressure xr.DataArray from the 'lev' dimension.
        Args:
        - refda (xr.DataArray): reference DataArray containing 'lev'
        Returns:
        - xr.DataArray: pressure DataArray
        '''
        p = refda.lev.expand_dims({'lat':refda.lat,'lon':refda.lon,'time':refda.time}).transpose('lat','lon','lev','time')
        return p

    def resample(self,da):
        '''
        Purpose: Compute a centered hourly mean (uses the two half-hour samples that straddle each hour; 
        falls back to one at boundaries).
        Args:
        - da (xr.DataArray): input DataArray
        Returns:
        - xr.DataArray: DataArray resampled at on-the-hour timestamps
        '''
        da = da.rolling(time=2,center=True,min_periods=1).mean()
        da = da.sel(time=da.time.dt.minute==0)
        return da

    def regrid(self,da):
        '''
        Purpose: Regrid a DataArray to 1.0° × 1.0° grid extending one grid cell beyond the target domain.
        Args:
        - da (xr.DataArray): input DataArray with radius (for better interpolation)
        Returns:
        - xr.DataArray: regridded DataArray
        '''
        targetlats = np.arange(self.latrange[0]-1.0,self.latrange[1]+2.0,1.0)
        targetlons = np.arange(self.lonrange[0]-1.0,self.lonrange[1]+2.0,1.0)
        targetgrid = xr.Dataset({'lat':(['lat'],targetlats),'lon':(['lon'],targetlons)})
        regridder  = xesmf.Regridder(da,targetgrid,method='conservative')
        da = regridder(da,keep_attrs=True)
        return da

    def calc_es(self,t):
        '''
        Purpose: Calculate saturation vapor pressure (eₛ) using Eqs. 17 and 18 from Huang J. (2018), J. Appl. 
        Meteorol. Climatol.
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

    def calc_qs(self,p,t):
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
        es = self.calc_es(t)
        qs = (epsilon*es)/(p-es*(1.0-epsilon))
        return qs

    def calc_rh(self,p,t,q):
        '''
        Purpose: Calculate relative humidity (RH) using qₛ.
        Args:
        - p (xr.DataArray): pressure DataArray (hPa)
        - t (xr.DataArray): temperature DataArray (K)
        - q (xr.DataArray): specific humidity DataArray (kg/kg)
        Returns:
        - xr.DataArray: RH DataArray (%)
        '''
        qs = self.calc_qs(p,t)
        rh = (q/qs)*100.0
        rh = rh.clip(min=0.0,max=100.0)
        return rh

    def calc_thetae(self,p,t,q=None):
        '''
        Purpose: Calculate (unsaturated or saturated) equivalent potential temperature (θₑ) using Eqs. 43 and 55 
        from Bolton D. (1980), Mon. Wea. Rev.     
        Args:
        - p (xr.DataArray): pressure DataArray (hPa)
        - t (xr.DataArray): temperature DataArray (K)
        - q (xr.DataArray, optional): specific humidity DataArray (kg/kg); if None, saturated θₑ computed
        Returns:
        - xr.DataArray: unsaturated or saturated θₑ DataArray (K)
        '''
        if q is None:
            q = self.calc_qs(p,t)
        p0 = 1000.0
        rv = 461.5
        rd = 287.04
        epsilon = rd/rv
        r  = q/(1.0-q)
        e  = (p*r)/(epsilon+r)
        tl = 2840.0/(3.5*np.log(t)-np.log(e)-4.805)+55.0
        thetae = t*(p0/p)**(0.2854*(1.0-0.28*r))*np.exp((3.376/tl-0.00254)*1000.0*r*(1.0+0.81*r))
        return thetae

    def calc_quadrature_weights(self,refda,rearth=6.371e6):
        '''
        Purpose: Compute separable quadrature weights for numerical integration over a 4D grid. For each dimension 
        ('lat', 'lon', 'lev', 'time'), centered finite differences estimate the spacing between adjacent grid points. 
        These spacings are combined into area weights ΔA (m²) accounting for spherical geometry, vertical thickness 
        weights Δp (hPa) for pressure levels, and temporal weights Δt (s) for constant-cadence time steps.
        Args:
        - refda (xr.DataArray): reference DataArray with dimensions 'lat', 'lon', 'lev', and 'time'
        - rearth (float): Earth's radius in meters (defaults to 6,371,000 m)
        Returns:
        - tuple(xr.DataArray, xr.DataArray, xr.DataArray): quadrature weights for ΔA (m²), Δp (hPa), and Δt (s)
        '''
        lats  = np.deg2rad(refda.lat.values)
        lons  = np.deg2rad(refda.lon.values)
        levs  = refda.lev.values
        times = refda.time.values
        dlat  = np.abs(np.concatenate([[lats[1]-lats[0]],0.5*(lats[2:]-lats[:-2]),[lats[-1]-lats[-2]]]))
        dlon  = np.abs(np.concatenate([[lons[1]-lons[0]],0.5*(lons[2:]-lons[:-2]),[lons[-1]-lons[-2]]]))
        dareavalues = (rearth**2*np.cos(lats)*dlat)[:,None]*dlon[None,:]
        dlevvalues  = np.abs(np.concatenate([[levs[1]-levs[0]],0.5*(levs[2:]-levs[:-2]),[levs[-1]-levs[-2]]]))
        dtimevalues = np.full(times.size,float(np.median(np.diff(times).astype('timedelta64[s]').astype(float))))
        darea = xr.DataArray(dareavalues.astype(np.float32),dims=('lat','lon'),coords={'lat':refda.lat,'lon':refda.lon})
        dlev  = xr.DataArray(dlevvalues.astype(np.float32),dims=('lev',),coords={'lev':refda.lev})
        dtime = xr.DataArray(dtimevalues.astype(np.float32),dims=('time',),coords={'time':refda.time})
        return darea,dlev,dtime

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
        ds.attrs = dict(history=f'Created on {datetime.today().strftime("%Y-%m-%d")} by {self.author} ({self.email})')
        logger.info(f'   {longname} size: {ds.nbytes*1e-9:.3f} GB')
        return ds

    def save(self,ds,timechunksize=2208):
        '''
        Purpose: Save an xr.Dataset to NetCDF and verify by reopening.
        Args:
        - ds (xr.Dataset): Dataset to save
        - timechunksize (int): chunk size for time dimension (defaults to 2,208 for 3-month chunks)
        Returns:
        - bool: True if save successful, False otherwise
        '''
        os.makedirs(self.savedir,exist_ok=True)
        shortname = list(ds.data_vars)[0]
        filename  = f'{shortname}.nc'
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