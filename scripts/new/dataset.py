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
        - maxlevs (int): maximum number of vertical levels to include; if maxlevs ≥ total levels, use all levels
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
            patchlength  = timemax-timemin
            neededlength = self.patchconfig.timelag+1
        else:
            timemin = timeidx
            timemax = timeidx+1
            patchlength = neededlength = 1
        patch = self.field[:,latmin:latmax,lonmin:lonmax,levmin:levmax,timemin:timemax]
        if patchlength<neededlength:
            padlength = neededlength-patchlength
            padshape  = (patch.shape[0],patch.shape[1],patch.shape[2],patch.shape[3],padlength)
            pad   = torch.zeros(padshape,dtype=self.fielddata.dtype)
            patch = torch.cat([pad,patch],dim=-1)
        sample = {
            'patch':patch,
            'target':self.target[latidx,lonidx,timeidx]}
        if self.uselocal and self.local is not None:
            sample['local'] = self.local[:,latidx,lonidx,timeidx]
        return sample