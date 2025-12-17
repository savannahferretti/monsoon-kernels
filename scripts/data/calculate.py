#!/usr/bin/env python

import logging
import warnings
from scripts.utils import Config
from scripts.data.classes import DataCalculator

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

if __name__=='__main__':
    config = Config()
    calculator = DataCalculator(
        author=config.author,
        email=config.email,
        filedir=config.rawdir,
        savedir=config.interimdir,
        latrange=config.latrange,
        lonrange=config.lonrange)

    logger.info('Importing all raw variables...')
    ps  = calculator.retrieve('ERA5_surface_pressure')
    t   = calculator.retrieve('ERA5_air_temperature')
    q   = calculator.retrieve('ERA5_specific_humidity')
    lf  = calculator.retrieve('ERA5_land_fraction')
    lhf = calculator.retrieve('ERA5_mean_surface_latent_heat_flux')
    shf = calculator.retrieve('ERA5_mean_surface_sensible_heat_flux')
    pr  = calculator.retrieve('IMERG_V06_precipitation_rate')

    logger.info('Resampling/regridding variables...')
    ps  = calculator.regrid(ps).load()
    t   = calculator.regrid(t).load()
    q   = calculator.regrid(q).load()
    lf  = calculator.regrid(lf).load()
    lhf = calculator.regrid(lhf).load()
    shf = calculator.regrid(shf).load()
    pr  = calculator.regrid(calculator.resample(pr)).clip(min=0).load()

    logger.info('Calculating relative humidity and equivalent potential temperature terms...')
    p          = calculator.create_p_array(q)
    rh         = calculator.calc_rh(p,t,q)
    thetae     = calculator.calc_thetae(p,t,q)
    thetaestar = calculator.calc_thetae(p,t)

    logger.info('Calculating quadrature weights...')
    darea,dlev,dtime = calculator.calc_quadrature_weights(t)

    logger.info('Creating datasets...')
    dslist = [
        calculator.create_dataset(t,'t','Air temperature','K'),
        calculator.create_dataset(q,'q','Specific humidity','kg/kg'),
        calculator.create_dataset(rh,'rh','Relative humidity','%'),
        calculator.create_dataset(thetae,'thetae','Equivalent potential temperature','K'),
        calculator.create_dataset(thetaestar,'thetaestar','Saturated equivalent potential temperature','K'),
        calculator.create_dataset(lf,'lf','Land fraction','0-1'),
        calculator.create_dataset(lhf,'lhf','Mean surface latent heat flux','W/m²'),
        calculator.create_dataset(shf,'shf','Mean surface sensible heat flux','W/m²'),
        calculator.create_dataset(pr,'pr','Precipitation rate','mm/hr'),
        calculator.create_dataset(darea,'darea','Horizontal area weights','m²'),
        calculator.create_dataset(dlev,'dlev','Vertical thickness weights','Pa'),
        calculator.create_dataset(dtime,'dtime','Time step weights (constant cadence)','s')]

    logger.info('Saving datasets...')
    for ds in dslist:
        calculator.save(ds)
        del ds