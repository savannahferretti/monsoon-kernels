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
    config = Config()
    downloader = DataDownloader(
        author=config.author,
        email=config.email,
        savedir=config.rawdir,
        latrange=config.latrange,
        lonrange=config.lonrange,
        levrange=config.levrange,
        years=config.years,
        months=config.months)
    downloader.download_all()