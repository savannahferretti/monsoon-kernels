#!/usr/bin/env python

import logging
import warnings
from scripts.utils import Config
from scripts.data.classes import DataSplitter

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

if __name__=='__main__':
    config = Config()
    splitter = DataSplitter(
        filedir=config.interimdir,
        savedir=config.splitsdir,
        trainrange=config.trainrange,
        validrange=config.validrange,
        testrange=config.testrange)
    splitter.split_all()
