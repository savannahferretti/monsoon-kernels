#!/usr/bin/env python

import logging
import warnings
from scripts.utils import Config
from scripts.data.classes import DataDownloader

logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

if __name__=='__main__':
    config     = Config()
    downloader = DataDownloader(config.author,config.email,config.rawdir,config.latrange,config.lonrange,config.levrange,config.years,config.months)
    logger.info('Retrieving ERA5 and IMERG data...')
    era5  = downloader.retrieve_era5()
    imerg = downloader.retrieve_imerg()
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
        downloader.process(psdata,'ps','ERA5 surface pressure','hPa',radius=4),
        downloader.process(tdata,'t','ERA5 air temperature','K',radius=4),
        downloader.process(qdata,'q','ERA5 specific humidity','kg/kg',radius=4),
        downloader.process(lfdata,'lf','ERA5 land fraction','0-1',radius=4),
        downloader.process(lhfdata,'lhf','ERA5 mean surface latent heat flux','W/m²',radius=4),
        downloader.process(shfdata,'shf','ERA5 mean surface sensible heat flux','W/m²',radius=4),
        downloader.process(prdata,'pr','IMERG V06 precipitation rate','mm/hr',radius=10)]
    del psdata,tdata,qdata,lfdata,lhfdata,shfdata,prdata
    logger.info('Saving datasets...')
    for ds in dslist:
        downloader.save(ds)
        del ds
