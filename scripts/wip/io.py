#!/usr/bin/env python

import os
import json
import torch
import logging
import xarray as xr

logger = logging.getLogger(__name__)

class IO:

    @staticmethod
    def get_split(split,filedir):
        '''
        Purpose: Retrieve a normalized data split from an HDF5 file.
        Args:
        - split (str): 'train' | 'valid' | 'test'
        - filedir (str): directory containing split files
        Returns:
        - xr.Dataset: split Dataset containing all variables
        '''
        filename = f'{split}.h5'
        filepath = os.path.join(filedir,filename)
        return xr.open_dataset(filepath,engine='h5netcdf')

    @staticmethod
    def get_stats(filedir):
        '''
        Purpose: Load normalization statistics from a JSON file (for denormalizing predictions).
        Args:
        - filedir (str): directory containing 'stats.json'
        Returns:
        - dict: statistics dictionary
        '''
        filename = 'stats.json'
        filepath = os.path.join(filedir,filename)
        with open(filepath,'r',encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def get_model(model,name,filedir):
        '''
        Purpose: Load trained model weights from a PyTorch checkpoint file into the provided model instance.
        Args:
        - model (torch.nn.Module): initialized model instance
        - name (str): model name
        - filedir (str): directory containing model checkpoints
        Returns:
        - bool: True if loading successful, False otherwise
        '''
        filename = f'{name}.pth'
        filepath = os.path.join(filedir,filename)
        if not os.path.exists(filepath):
            logger.error(f'Checkpoint not found: {filepath}')
            return False
        state = torch.load(filepath,map_location='cpu')
        model.load_state_dict(state)
        logger.info(f'   Loaded checkpoint from {filename}')
        return True

    @staticmethod
    def save_model(state,name,savedir):
        '''
        Purpose: Save a model state dictionary to a PyTorch checkpoint file, then verify by reopening.
        Args:
        - state (dict): model state dictionary containing trained weights
        - name (str): model name
        - savedir (str): output directory
        Returns:
        - bool: True if save and verification successful, False otherwise
        '''
        os.makedirs(savedir,exist_ok=True)
        filename = f'{name}.pth'
        filepath = os.path.join(savedir,filename)
        logger.info(f'   Attempting to save {filename}...')
        try:
            torch.save(state,filepath)
            _ = torch.load(filepath,map_location='cpu')
            logger.info('      File write successful')
            return True
        except Exception:
            logger.exception('      Failed to save or verify')
            return False
    
    @staticmethod
    def save_predictions(arr,refda,name,split,savedir):
        '''
        Purpose: Save precipitation predictions to a NetCDF file on the target grid, then verify by reopening.
        Args:
        - arr (np.ndarray): denormalized predictions with shape (nlats, nlons, ntimes)
        - refda (xr.DataArray): reference DataArray providing target coordinates and dimensions
        - name (str): model name
        - split (str): 'valid' | 'test'
        - savedir (str): output directory
        Returns:
        - bool: True if save and verification successful, False otherwise
        '''
        da = xr.DataArray(arr,dims=refda.dims,coords=refda.coords,name='pr')
        da.attrs = dict(long_name='NN-predicted precipitation rate',units='mm/hr')
        os.makedirs(savedir,exist_ok=True)
        filename = f'{name}_{split}_pr.nc'
        filepath = os.path.join(savedir,filename)
        logger.info(f'   Attempting to save {filename}...')
        try:
            da.to_netcdf(filepath,engine='h5netcdf')
            with xr.open_dataset(filepath,engine='h5netcdf') as _:
                pass
            logger.info('      File write successful')
            return True
        except Exception:
            logger.exception('      Failed to save or verify')
            return False