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
        - author (str): author name for metadata
        - email (str): author email for metadata
        - filedir (str): directory containing raw NetCDF files
        - savedir (str): directory to save processed files
        - latrange (tuple[float,float]): target latitude range
        - lonrange (tuple[float,float]): target longitude range
        '''
        self.author   = author
        self.email    = email
        self.filedir  = filedir
        self.savedir  = savedir
        self.latrange = latrange
        self.lonrange = lonrange

    def retrieve(self,longname):
        '''
        Purpose: Lazily import a NetCDF file as a DataArray with ascending pressure levels.
        Args:
        - longname (str): variable description
        Returns:
        - xr.DataArray: DataArray with levels ordered if applicable
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
        Purpose: Create a pressure DataArray from the lev dimension.
        Args:
        - refda (xr.DataArray): reference DataArray containing lev
        Returns:
        - xr.DataArray: pressure DataArray
        '''
        p = refda.lev.expand_dims({'lat':refda.lat,'lon':refda.lon,'time':refda.time}).transpose('lat','lon','lev','time')
        return p

    def resample(self,da):
        '''
        Purpose: Compute centered hourly mean using two half-hour samples.
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
        Purpose: Regrid a DataArray to 1.0° × 1.0° grid extending 1 cell beyond domain.
        Args:
        - da (xr.DataArray): input DataArray with radius
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
        Purpose: Calculate saturation vapor pressure using Huang (2018) equations.
        Args:
        - t (xr.DataArray): temperature DataArray in K
        Returns:
        - xr.DataArray: saturation vapor pressure in hPa
        '''
        tc = t-273.15
        eswat = np.exp(34.494-(4924.99/(tc+237.1)))/((tc+105.0)**1.57)
        esice = np.exp(43.494-(6545.8/(tc+278.0)))/((tc+868.0)**2.0)
        es = xr.where(tc>0.0,eswat,esice)/100.0
        return es

    def calc_qs(self,p,t):
        '''
        Purpose: Calculate saturation specific humidity using Miller (2018) equation.
        Args:
        - p (xr.DataArray): pressure DataArray in hPa
        - t (xr.DataArray): temperature DataArray in K
        Returns:
        - xr.DataArray: saturation specific humidity in kg/kg
        '''
        rv = 461.50
        rd = 287.04
        epsilon = rd/rv
        es = self.calc_es(t)
        qs = (epsilon*es)/(p-es*(1.0-epsilon))
        return qs

    def calc_rh(self,p,t,q):
        '''
        Purpose: Calculate relative humidity using saturation specific humidity.
        Args:
        - p (xr.DataArray): pressure DataArray in hPa
        - t (xr.DataArray): temperature DataArray in K
        - q (xr.DataArray): specific humidity DataArray in kg/kg
        Returns:
        - xr.DataArray: relative humidity in %
        '''
        qs = self.calc_qs(p,t)
        rh = (q/qs)*100.0
        rh = rh.clip(min=0.0,max=100.0)
        return rh

    def calc_thetae(self,p,t,q=None):
        '''
        Purpose: Calculate equivalent potential temperature using Bolton (1980) equations.
        Args:
        - p (xr.DataArray): pressure DataArray in hPa
        - t (xr.DataArray): temperature DataArray in K
        - q (xr.DataArray, optional): specific humidity in kg/kg; if None, saturated θₑ computed
        Returns:
        - xr.DataArray: equivalent potential temperature in K
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
        Purpose: Compute separable quadrature components for a 4D grid.
        Args:
        - refda (xr.DataArray): reference DataArray with lat, lon, lev, time
        - rearth (float): Earth radius in meters (defaults to 6,371,000)
        Returns:
        - tuple(xr.DataArray,xr.DataArray,xr.DataArray): horizontal area, vertical thickness, time step weights
        '''
        dims  = ('lat','lon','lev','time')
        refda = refda.transpose(*dims)
        lats  = refda.lat.values
        lons  = refda.lon.values
        levs  = refda.lev.values*100.0
        times = refda.time.values
        def spacing(coord):
            '''
            Purpose: Estimate spacing using centered differences.
            Args:
            - coord (np.ndarray): 1D monotonically increasing coordinates
            Returns:
            - np.ndarray: spacing between grid points
            '''
            coord = np.asarray(coord,dtype=np.float64)
            delta = np.empty_like(coord)
            delta[1:-1]   = 0.5*(coord[2:]-coord[:-2])
            delta[[0,-1]] = (coord[1]-coord[0],coord[-1]-coord[-2])
            return np.abs(delta)
        dlat  = spacing(np.deg2rad(lats))
        dlon  = spacing(np.deg2rad(lons))
        dareavalues = ((rearth**2)*np.cos(np.deg2rad(lats))*dlat).reshape(lats.size,1)*dlon.reshape(1,lons.size)
        dlevvalues  = spacing(levs)
        dtimescalar = float(np.nanmedian((np.diff(times)/np.timedelta64(1,'s')).astype(np.float64)))
        dtimevalues = np.full(times.size,dtimescalar)
        darea = xr.DataArray(dareavalues.astype(np.float32),dims=('lat','lon'),coords={'lat':refda.lat,'lon':refda.lon})
        dlev  = xr.DataArray(dlevvalues.astype(np.float32),dims=('lev',),coords={'lev':refda.lev})
        dtime = xr.DataArray(dtimevalues.astype(np.float32),dims=('time',),coords={'time':refda.time})
        return darea,dlev,dtime

    def create_dataset(self,da,shortname,longname,units):
        '''
        Purpose: Wrap a DataArray into a Dataset with metadata.
        Args:
        - da (xr.DataArray): input DataArray
        - shortname (str): variable name
        - longname (str): variable description
        - units (str): variable units
        Returns:
        - xr.Dataset: Dataset with variable and metadata
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
        Purpose: Save a Dataset to NetCDF and verify by reopening.
        Args:
        - ds (xr.Dataset): Dataset to save
        - timechunksize (int): chunk size for time dimension (defaults to 2208)
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

    def calculate_all(self):
        '''
        Purpose: Calculate and process all derived variables.
        Returns:
        - bool: True if all calculations successful, False otherwise
        '''
        logger.info('Importing all raw variables...')
        ps  = self.retrieve('ERA5_surface_pressure')
        t   = self.retrieve('ERA5_air_temperature')
        q   = self.retrieve('ERA5_specific_humidity')
        lf  = self.retrieve('ERA5_land_fraction')
        lhf = self.retrieve('ERA5_mean_surface_latent_heat_flux')
        shf = self.retrieve('ERA5_mean_surface_sensible_heat_flux')
        pr  = self.retrieve('IMERG_V06_precipitation_rate')
        logger.info('Resampling/regridding variables...')
        ps  = self.regrid(ps).load()
        t   = self.regrid(t).load()
        q   = self.regrid(q).load()
        lf  = self.regrid(lf).load()
        lhf = self.regrid(lhf).load()
        shf = self.regrid(shf).load()
        pr  = self.regrid(self.resample(pr)).clip(min=0).load()
        logger.info('Calculating relative humidity and equivalent potential temperature terms...')
        p          = self.create_p_array(q)
        rh         = self.calc_rh(p,t,q)
        thetae     = self.calc_thetae(p,t,q)
        thetaestar = self.calc_thetae(p,t)
        logger.info('Calculating quadrature weights...')
        darea,dlev,dtime = self.calc_quadrature_weights(t)
        logger.info('Creating datasets...')
        dslist = [
            self.create_dataset(t,'t','Air temperature','K'),
            self.create_dataset(q,'q','Specific humidity','kg/kg'),
            self.create_dataset(rh,'rh','Relative humidity','%'),
            self.create_dataset(thetae,'thetae','Equivalent potential temperature','K'),
            self.create_dataset(thetaestar,'thetaestar','Saturated equivalent potential temperature','K'),
            self.create_dataset(lf,'lf','Land fraction','0-1'),
            self.create_dataset(lhf,'lhf','Mean surface latent heat flux','W/m²'),
            self.create_dataset(shf,'shf','Mean surface sensible heat flux','W/m²'),
            self.create_dataset(pr,'pr','Precipitation rate','mm/hr'),
            self.create_dataset(darea,'darea','Horizontal area weights','m²'),
            self.create_dataset(dlev,'dlev','Vertical thickness weights','Pa'),
            self.create_dataset(dtime,'dtime','Time step weights (constant cadence)','s')]
        logger.info('Saving datasets...')
        success = True
        for ds in dslist:
            if not self.save(ds):
                success = False
            del ds
        return success
