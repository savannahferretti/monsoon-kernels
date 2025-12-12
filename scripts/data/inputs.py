#!/usr/bin/env python

import os 
import torch 
import numpy as np 
import xarray as xr  
from patch import PatchGeometry,PatchDataset 

class InputDataModule:

    @staticmethod
    def prepare(splits,fieldvars,localvars,targetvar,filedir):
        '''
        Purpose: Convert variable xr.DataArrays into PyTorch tensors by data type and extract quadrature weights and coordinates.
        Args:
        - splits (list[str]): list of data splits to load
        - fieldvars (list[str]): predictor field variable names
        - localvars (list[str]): local input variable names
        - targetvar (str): target variable name
        - filedir (str): directory containing split files
        Returns:
        - dict[str,dict]: dictionary mapping split names to data dictionaries containing PyTorch tensors and coordinates
        '''
        result = {}
        for split in splits:
            filename = f'{split}.h5'
            filepath = os.path.join(filedir,filename)
            ds = xr.open_dataset(filepath,engine='h5netcdf')
            field  = torch.from_numpy(np.stack([ds[varname].values for varname in fieldvars],axis=0))
            local  = (torch.from_numpy(np.stack([ds[varname].values if 'time' in ds[varname].dims else ds[varname].expand_dims(time=ds.time).values
                                                 for varname in localvars],axis=0))) if localvars else None
            target = torch.from_numpy(ds[targetvar].values)
            quad   = torch.from_numpy(ds['quad'].values)
            result[split] = { 
                'refda':ds[targetvar],
                'field':field,
                'local':local,
                'target':target,
                'quad':quad,
                'lats':ds.lat.values,
                'lons':ds.lon.values}
        return result        

    @staticmethod
    def dataloaders(splitdata,patchconfig,uselocal,latrange,lonrange,batchsize,workers,device):
        '''
        Purpose: Build PatchGeometry, centers, PatchDatasets, and DataLoaders for given splits.
        Args:
        - splitdata (dict): dictionary from prepare()
        - patchconfig (dict): patch configuration
        - uselocal (bool): whether to use local inputs
        - latrange (tuple[float,float]): latitude range 
        - lonrange (tuple[float,float]): longitude range 
        - batchsize (int): batch size for PyTorch DataLoader
        - workers (int): number of PyTorch DataLoader workers
        - device (str): device to use
        Returns:
        - dict[str,object]: dictionary containing the patch geometry, valid patch centers, constructed datasets, 
          and dataloaders
        '''
        geometry = PatchGeometry(patchconfig['radius'],patchconfig['maxlevs'],patchconfig['timelag'])
        kwargs   = dict(num_workers=workers,pin_memory=(device=='cuda'),persistent_workers=(workers>0))
        if workers>0:
            kwargs['prefetch_factor'] = 2
        centers  = {}
        datasets = {}
        loaders  = {}
        for split,data in splitdata.items():
            centers[split]  = geometry.centers(data['target'],data['lats'],data['lons'],latrange,lonrange)
            datasets[split] = PatchDataset(geometry,centers[split],data['field'],data['quad'],data['local'],data['target'],uselocal)
            loaders[split]  = torch.utils.data.DataLoader(datasets[split],batch_size=batchsize,shuffle=(split=='train'),**kwargs)
        return {
            'geometry':geometry,
            'centers':centers,
            'datasets':datasets,
            'loaders':loaders}