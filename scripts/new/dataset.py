#!/usr/bin/env python

import os
import torch
import numpy as np
import xarray as xr

class PatchGeometry:
    
    def __init__(self,radius,maxlevs,timelag):
        '''
        Purpose: Initialize patch geometry to infer patch shape and valid patch centers.
        Args:
        - radius (int): number of horizontal grid points to include on each side of the center point
        - maxlevs (int): maximum number of vertical levels to include; should be ≤ total levels
        - timelag (int): number of past time steps to include; if 0, use only the current time step (no time lag)
        '''
        self.radius = int(radius)
        self.maxlevs = int(maxlevs)
        self.timelag = int(timelag)
    
    def shape(self):
        '''
        Purpose: Infer (plats, plons, plevs, ptimes) from patch geometry and grid size.
        Returns:
        - tuple[int,int,int,int]: (plats, plons, plevs, ptimes)
        '''
        plats  = 2*self.radius+1
        plons  = 2*self.radius+1
        plevs  = self.maxlevs
        ptimes = self.timelag+1 if self.timelag>0 else 1
        return (plats,plons,plevs,ptimes)
    
    def centers(self,target,lats,lons,latrange,lonrange):
        '''
        Purpose: Build (latidx, lonidx, timeidx) centers where the patch fits, target is finite, and (lat, lon) lies 
        inside the prediction domain.
        Args:
        - target (torch.Tensor): target values with shape (nlats, nlons, ntimes)
        - lats (np.ndarray): latitude values with shape (nlats,)
        - lons (np.ndarray): longitude values with shape (nlons,)
        - latrange (tuple[float,float]): latitude range
        - lonrange (tuple[float,float]): longitude range
        Returns:
        - list[tuple[int,int,int]]: list of (latidx, lonidx, timeidx) patch centers
        '''
        nlats,nlons,ntimes       = target.shape
        latidxs,lonidxs,timeidxs = torch.nonzero(torch.isfinite(target),as_tuple=True)
        lats = torch.as_tensor(lats,dtype=torch.float32)
        lons = torch.as_tensor(lons,dtype=torch.float32)
        patchfits = ((latidxs>=self.radius)&(latidxs<nlats-self.radius)&(lonidxs>=self.radius)&(lonidxs<nlons-self.radius))
        indomain  = ((lats[latidxs]>=latrange[0])&(lats[latidxs]<=latrange[1])&(lons[lonidxs]>=lonrange[0])&(lons[lonidxs]<=lonrange[1]))
        validlatidxs  = latidxs[patchfits&indomain].tolist()
        validlonidxs  = lonidxs[patchfits&indomain].tolist()
        validtimeidxs = timeidxs[patchfits&indomain].tolist()
        return list(zip(validlatidxs,validlonidxs,validtimeidxs))

class PatchDataset(torch.utils.data.Dataset):
    
    def __init__(self,field,local,target,centers,geometry,uselocal):
        '''
        Purpose: Return predictor field patches, optional local inputs, and target values for (lat, lon, time) samples.
        Args:
        - field (torch.Tensor): predictor fields with shape (nfieldvars, nlats, nlons, nlevs, ntimes)
        - local (torch.Tensor | None): local inputs with shape (nlocalvars, nlats, nlons, ntimes)
        - target (torch.Tensor): target values with shape (nlats, nlons, ntimes)
        - centers (list[tuple[int,int,int]]): (latidx, lonidx, timeidx) patch centers
        - geometry (PatchGeometry): patch geometry
        - uselocal (bool): whether to use local inputs
        '''
        super().__init__()
        if field.ndim != 5:
            raise ValueError('`field` must have shape (nfieldvars, nlats, nlons, nlevs, ntimes)')
        if local is not None and local.ndim != 4:
            raise ValueError('`local` must have shape (nlocalvars, nlats, nlons, ntimes) or be None')
        if target.ndim != 3:
            raise ValueError('`target` must have shape (nlats, nlons, ntimes)')
        self.field    = field
        self.local    = local
        self.target   = target
        self.centers  = list(centers)
        self.geometry = geometry
        self.uselocal = uselocal
    
    def __len__(self):
        '''
        Purpose: Return the number of valid centers in the dataset.
        Returns:
        - int: number of centers
        '''
        return len(self.centers)

    def __getitem__(self,idx):
        '''
        Purpose: Extract a single sample containing a predictor fields patch, optional local inputs, and a target value.
        Args:
        - centeridx (int): index into valid centers list
        Returns:
        - dict: sample with keys 'patch', 'target', and optionally 'local'
        '''
        latidx,lonidx,timeidx      = self.centers[idx]
        _,nlats,nlons,nlevs,ntimes = self.field.shape
        latmin,latmax = latidx-self.geometry.radius,latidx+self.geometry.radius+1
        lonmin,lonmax = lonidx-self.geometry.radius,lonidx+self.geometry.radius+1
        levmin,levmax = 0,self.geometry.maxlevs
        if self.geometry.timelag>0:
            timemin,timemax = max(0,timeidx-self.geometry.timelag),timeidx+1
            patchtimelength = timemax-timemin
            neededlength    = self.geometry.timelag+1
        else:
            timemin,timemax = timeidx,timeidx+1
            patchtimelength = neededlength = 1
        patch = self.field[:,latmin:latmax,lonmin:lonmax,levmin:levmax,timemin:timemax]
        if patchtimelength<neededlength:
            padlength = neededlength-patchtimelength
            padshape  = (patch.shape[0],patch.shape[1],patch.shape[2],patch.shape[3],padlength)
            pad   = torch.zeros(padshape,dtype=self.field.dtype)
            patch = torch.cat([pad,patch],dim=-1)
        sample = {
            'patch':patch,
            'target':self.target[latidx,lonidx,timeidx]}
        if self.uselocal and self.local is not None:
            sample['local'] = self.local[:,latidx,lonidx,timeidx]
        return sample

class DataModule:

    @staticmethod
    def prepare(splits,fieldvars,localvars,targetvar,filedir):
        '''
        Purpose: Retrieve data splits as xr.Datasets, convert variable xr.DataArrays into to PyTorch tensors by data type, and extract 
        quadrature weights and coordinates.
        Args:
        - splits (list[str]): list of splits to load
        - fieldvars (list[str]): predictor field variable names
        - localvars (list[str]): local input variable names
        - targetvar (str): target variable name
        - filedir (str): directory containing split files
        Returns:
        - dict[str,dict]: dictionary mapping split names to data dictionaries containing tensors and coordinates
        '''
    result = {}
    for split in splits:
        filename = f'{split}.h5'
        filepath = os.path.join(filedir,filename)
        ds = xr.open_dataset(filepath,engine='h5netcdf')
        fieldlist = []
        for varname in fieldvars:
            da  = ds[varname]
            arr = da.values
            fieldlist.append(arr)
        field = torch.from_numpy(np.stack(fieldlist,axis=0))
        if localvars:
            locallist = []
            for varname in localvars:
                da  = ds[varname]
                arr = da.values if 'time' in da.dims else np.broadcast_to(da.values[...,None],(da.values.shape[0],da.values.shape[1],ds.time.size))
                locallist.append(arr)
            local = torch.from_numpy(np.stack(locallist,axis=0))
        else:
            local = None
        target = torch.from_numpy(ds[targetvar].values)
        quad   = torch.from_numpy(ds['quad'].values)
        result[split] = { 
            'ds':ds,
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
        - splitdata (dict): dictionary from prepare_splits() containing tensors and coordinates
        - patchconfig (dict): patch configuration with keys 'radius', 'maxlevs', and 'timelag'
        - uselocal (bool): whether to use local inputs
        - latrange (tuple[float,float]): latitude range 
        - lonrange (tuple[float,float]): longitude range 
        - batchsize (int): batch size for DataLoader
        - workers (int): number of DataLoader workers
        - device (str): device to use
        Returns:
        - dict: dictionary containing patch geometry, centers, datasets, loaders, and quadrature weights
        '''
        geometry     = PatchGeometry(patchconfig['radius'],patchconfig['maxlevs'],patchconfig['timelag'])
        commonkwargs = dict(num_workers=workers,pin_memory=(device=='cuda'),persistent_workers=(workers>0))
        if workers>0:
            commonkwargs['prefetch_factor'] = 2
        centers  = {}
        datasets = {}
        loaders  = {}
        quad = None
        for split,data in splitdata.items():
            centers[split]  = geometry.centers(data['target'],data['lats'],data['lons'],latrange,lonrange)
            datasets[split] = PatchDataset(data['field'],data['local'],data['target'],centers[split],geometry,uselocal)
            loaders[split]  = torch.utils.data.DataLoader(datasets[split],batch_size=batchsize,shuffle=(split=='train'),**commonkwargs)
            if quad is None:
                quad = data['quad']
        return {
            'geometry':geometry,
            'centers':centers,
            'datasets':datasets,
            'loaders':loaders,
            'quad':quad}