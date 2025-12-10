#!/usr/bin/env python

import os
import torch
import numpy as np
import xarray as xr

class PatchConfig:
    
    def __init__(self,radius,maxlevs,timelag):
        '''
        Purpose: Store patch configuration and infer patch shape/valid patch centers.
        Args:
        - radius (int): number of horizontal grid points to include on each side of the center point
        - maxlevs (int): maximum number of vertical levels to include; if 'maxlevs' ≥ total levels, use all levels
        - timelag (int): number of past time steps to include; if 0, use only the current time step (no time lag)
        '''
        self.radius = int(radius)
        self.maxlevs = int(maxlevs)
        self.timelag = int(timelag)
    
    def shape(self,nlevs):
        '''
        Purpose: Infer (plats, plons, plevs, ptimes) from patch configuration and grid size.
        Args:
        - nlevs (int): number of vertical levels in the full grid
        Returns:
        - tuple[int,int,int,int]: (plats, plons, plevs, ptimes)
        '''
        plats  = 2*self.radius+1
        plons  = 2*self.radius+1
        plevs  = min(nlevs,self.maxlevs)
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

class SampleDataset(torch.utils.data.Dataset):
    
    def __init__(self,field,local,target,centers,patchconfig,uselocal):
        '''
        Purpose: Return predictor field patches, optional local inputs, and target values for (lat, lon, time) samples.
        Args:
        - field (torch.Tensor): predictor fields with shape (nfieldvars, nlats, nlons, nlevs, ntimes)
        - local (torch.Tensor | None): local inputs with shape (nlocalvars, nlats, nlons, ntimes)
        - target (torch.Tensor): target values with shape (nlats, nlons, ntimes)
        - centers (list[tuple[int,int,int]]): (latidx, lonidx, timeidx) patch centers
        - patchconfig (Patch): patch configuration
        - uselocal (bool): whether to use local inputs
        '''
        super().__init__()
        if field.ndim != 5:
            raise ValueError('`field` must have shape (nfieldvars, nlats, nlons, nlevs, ntimes)')
        if local is not None and local.ndim != 4:
            raise ValueError('`local` must have shape (nlocalvars, nlats, nlons, ntimes) or be None')
        if target.ndim != 3:
            raise ValueError('`target` must have shape (nlats, nlons, ntimes)')
        self.field       = field
        self.local       = local
        self.target      = target
        self.centers     = list(centers)
        self.patchconfig = patchconfig
        self.uselocal    = uselocal
    
    def __len__(self):
        '''
        Purpose: Return the number of samples in the dataset.
        Returns:
        - int: number of samples
        '''
        return len(self.centers)

    def __getitem__(self,centeridx):
        '''
        Purpose: Extract a single sample containing a predictor fields patch, optional local inputs, and a target value.
        Args:
        - centeridx (int): (latidx, lonidx, timeidx) for the sample
        Returns:
        - dict: sample with keys 'patch', 'target', and optionally 'local'
        '''
        latidx,lonidx,timeidx      = self.centers[centeridx]
        _,nlats,nlons,nlevs,ntimes = self.field.shape
        latmin,latmax = latidx-self.patchconfig.radius,latidx+self.patchconfig.radius+1
        lonmin,lonmax = lonidx-self.patchconfig.radius,lonidx+self.patchconfig.radius+1
        levmin,levmax = 0,min(nlevs,self.patch.maxlevs)
        if self.patchconfig.timelag>0:
            timemin,timemax = max(0,timeidx-self.patchconfig.timelag),timeidx+1
            patchtimelength = timemax-timemin
            neededlength    = self.patchconfig.timelag+1
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

class DataPrep:

    @staticmethod
    def prepare(splits,fieldvars,localvars,targetvar,fieldir):
        '''
        Purpose: Retrieve data splits as xr.Datasets, convert variable xr.DataArrays into to PyTorch tensors by data type, and extract quadrature weights and coordinates.
        Args:
        - splits (list[str]): list of splits to load (e.g., ['train', 'valid'])
        - fieldvars (list[str]): predictor field variable names
        - localvars (list[str]): local input variable names
        - targetvar (str): target variable name
        - filedir (str): directory containing split files
        Returns:
        - dict: dictionary with 'ds', 'field', 'local', 'target', and 'quadweights'
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
        if localvars:
            locallist = []
            for varname in localvars:
                da  = ds[varname]
                arr = da.values if 'time' in da.dims else np.broadcast_to(da.values[...,None],(da.values.shape[0],da.values.shape[1],ds.time.size))
                locallist.append(arr)
            local = torch.from_numpy(np.stack(locallist,axis=0))
        else:
            local = None
            results[split] = {
                'ds':ds,
                'field':torch.from_numpy(np.stack(fieldlist,axis=0)),
                'local':local,
                'target':torch.from_numpy(ds[targetvar].values),
                'quadweights':torch.from_numpy(ds['quadweights'].values),
                'lats':ds.lat.values,
                'lons':ds.lon.values}
        return results

    @staticmethod
    def dataloaders(splitdata,patchconfig,uselocal,latrange,lonrange,batchsize,workers,device,nlevs):
        '''
        Purpose: Build PatchConfig, centers, SampleDatasets, and DataLoaders for given splits.
        Args:
        - splitdata (dict): dictionary from prepare_splits() containing tensors and coordinates
        - patchconfig (dict): patch configuration from model config
        - uselocal (bool): whether to use local inputs
        - latrange (tuple[float,float]): latitude range for domain filtering
        - lonrange (tuple[float,float]): longitude range for domain filtering
        - batchsize (int): batch size for DataLoader
        - workers (int): number of DataLoader workers
        - device (str): device for training ('cuda' | 'cpu')
        - nlevs (int): number of vertical levels
        Returns:
        - dict: dictionary with 'patch', 'patchshape', 'centers', 'datasets', 'loaders', and 'quadweights'
        '''
        patch        = PatchConfig(patchconfig['radius'],maxlevs=patchconfig['maxlevs'],patchconfig['timelag'])
        patchshape   = patch.shape(nlevs)
        commonkwargs = dict(num_workers=workers,pin_memory=(device=='cuda'),persistent_workers=(workers>0))
        if workers>0:
            commonkwargs['prefetch_factor'] = 2
        centers  = {}
        datasets = {}
        loaders  = {}
        quadweights = None
        for split,data in splitdata.items():
            centers[split]  = patch.centers(data['target'],data['lats'],data['lons'],latrange,lonrange)
            datasets[split] = SampleDataset(data['field'],data['local'],data['target'],centers[split],patch,uselocal)
            loaders[split]  = torch.util.data.DataLoader(datasets[splitname],batch_size=batchsize,shuffle=(split=='train'),**commonkwargs)
        return {
            'patch':patch,
            'patchshape':patchshape,
            'centers':centers,
            'datasets':datasets,
            'loaders':loaders,
            'quadweights':quadweights}