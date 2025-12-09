#!/usr/bin/env python

import torch
import numpy as np
from torch.utils.data import Dataset

class Data:

    @staticmethod
    def get_tensor(ds,fieldvars,localvars,targetvar):
        '''
        Purpose: Convert field, local, and target variables from an xr.Dataset to stacked PyTorch tensors.
        Args:
        - ds (xr.Dataset): split Dataset containing all variables
        - fieldvars (list[str]): predictor field variable names
        - localvars (list[str]): local input variable names
        - targetvar (str): target variable name
        Returns:
        - tuple[torch.Tensor,torch.Tensor | None,torch.Tensor]: field, local (or None), and target tensors
        '''
        fieldlist = []
        for varname in fieldvars:
            arr = ds[varname].values
            fieldlist.append(arr)
        field = torch.from_numpy(np.stack(fieldlist,axis=0))
        if localvars:
            locallist = []
            for varname in localvars:
                da = ds[varname]
                if 'time' in da.dims:
                    arr = ds[varname].values
                else:
                    arr = np.broadcast_to(ds[varname].values[...,None],(ds.lat.size,ds.lon.size,ds.time.size))
                locallist.append(arr)
            local = torch.from_numpy(np.stack(locallist,axis=0))
        else:
            local = None
        target = torch.from_numpy(ds[targetvar].values)
        return field,local,target

class Patch:
    
    def __init__(self,radius,maxlevs,timelag):
        '''
        Purpose: Initialize the space-height-time patch configuration.
        Args:
        - radius (int): number of horizontal grid points to include on each side of the center point
        - maxlevs (int): maximum number of vertical levels to include from the surface upward
        - timelag (int): number of past time steps to include (0 means current time step only, no lag)
        '''
        self.radius  = int(radius)
        self.maxlevs = int(maxlevs)
        self.timelag = int(timelag)
    
    def get_shape(self,nlevs):
        '''
        Purpose: Compute patch dimensions based on configuration and available vertical levels.
        Args:
        - nlevs (int): total number of vertical levels available in the full grid
        Returns:
        - tuple[int,int,int,int]: patch dimensions
        '''
        plats  = 2*self.radius+1
        plons  = 2*self.radius+1
        plevs  = min(nlevs,self.maxlevs)
        ptimes = self.timelag+1 if self.timelag>0 else 1
        return (plats,plons,plevs,ptimes)
    
    def get_centers(self,target,lats,lons,latrange,lonrange):
        '''
        Purpose: Identify valid (latidx, lonidx, timeidx) center locations where the target value is finite, the patch fits within grid 
        boundaries, and the location falls within the prediction domain.
        Args:
        - target (torch.Tensor): target array with shape (nlats, nlons, ntimes)
        - lats (np.ndarray): latitude coordinate values with shape (nlats,)
        - lons (np.ndarray): longitude coordinate values with shape (nlons,)
        - latrange (tuple[float,float]): minimum/maximum latitude for the prediction domain
        - lonrange (tuple[float,float]): minimum/maximum longitude for the prediction domain
        Returns:
        - list[tuple[int,int,int]]: list of valid center indices
        '''
        latidx,lonidx,timeidx = torch.nonzero(torch.isfinite(target),as_tuple=True)
        nlats,nlons,_ = target.shape
        lats,lons     = torch.as_tensor(lats,dtype=torch.float32),torch.as_tensor(lons,dtype=torch.float32)
        patchfits = ((latidx>=self.radius)&(latidx<nlats-self.radius)&(lonidx>=self.radius)&(lonidx<nlons-self.radius))
        indomain  = ((lats[latidx]>=latrange[0])&(lats[latidx]<=latrange[1])&(lons[lonidx]>=lonrange[0])&(lons[lonidx]<=lonrange[1]))
        latidxkeep  = latidx[patchfits&indomain].tolist()
        lonidxkeep  = lonidx[patchfits&indomain].tolist()
        timeidxkeep = timeidx[patchfits&indomain].tolist()
        return list(zip(latidxkeep,lonidxkeep,timeidxkeep))

class SampleDataset(Dataset):
    
    def __init__(self,field,local,target,centers,patch,uselocal):
        '''
        Purpose: Initialize dataset with full tensors and valid center locations for sampling.
        Args:
        - field (torch.Tensor): full predictor field data with shape (nfieldvars, nlats, nlons, nlevs, ntimes)
        - local (torch.Tensor | None): full local data with (nlocalvars, nlats, nlons, ntimes) or None
        - target (torch.Tensor): full target data with shape (nlats, nlons, ntimes)
        - centers (list[tuple[int,int,int]]): list of valid (latidx, lonidx, timeidx) center indices for sampling
        - patch (Patch): patch configuration object
        - uselocal (bool): whether to include local inputs in returned samples
        '''
        super().__init__()
        if field.ndim!=5 or (local is not None and local.ndim!=4) or target.ndim!=3:
            raise ValueError('Invalid tensor dimensions')
        self.field    = field
        self.local    = local
        self.target   = target
        self.centers  = list(centers)
        self.patch    = patch
        self.uselocal = uselocal
    
    def __len__(self):
        '''
        Purpose: Return the total number of valid samples available in this dataset.
        Returns:
        - int: number of valid centers
        '''
        return len(self.centers)
    
    def __getitem__(self,idx):
        '''
        Purpose: Extract a space-height-time patch centered at a valid location and its corresponding local and target values.
        Zero-pads temporal dimension if insufficient past time steps exist.
        Args: 
        - idx (int): sample index into the list of valid centers
        Returns:
        - dict: sample containing field patch, target scalar, and optional local inputs at center
        '''
        (latidx,lonidx,timeidx),nlevs = self.centers[idx],self.field.shape[3]
        latslice = slice(latidx-self.patch.radius,latidx+self.patch.radius+1)
        lonslice = slice(lonidx-self.patch.radius,lonidx+self.patch.radius+1)
        levslice = slice(0,min(nlevs,self.patch.maxlevs))
        if self.patch.timelag>0:
            timestart = max(0,timeidx-self.patch.timelag)
            patch  = self.field[:,latslice,lonslice,levslice,timestart:timeidx+1]
            needed = self.patch.timelag+1
            if patch.shape[-1]<needed:
                pad   = torch.zeros(*patch.shape[:-1],needed-patch.shape[-1],dtype=self.field.dtype)
                patch = torch.cat([pad,patch],dim=-1)
        else:
            patch = self.field[:,latslice,lonslice,levslice,timeidx:timeidx+1]
        sample = {'patch':patch,'target':self.target[latidx,lonidx,timeidx]}
        if self.uselocal and self.local is not None:
            sample['local'] = self.local[:,latidx,lonidx,timeidx]
        return sample