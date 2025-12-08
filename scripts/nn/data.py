#!/usr/bin/env python

import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset


def load(splitname,filedir):
    '''
    Purpose: Load a split as an xr.Dataset.
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - filedir (str): directory containing split files
    '''
    filename = f'{splitname}.h5'
    filepath = os.path.join(filedir,filename)
    ds = xr.open_dataset(filepath,engine='h5netcdf')
    return ds


def tensors(ds,fieldvars,localvars,targetvar):
    '''
    Purpose: Convert xr.Dataset for a given split to variable-wise torch.Tensor containers for SampleDataset.
    Args:
    - ds (xr.Dataset): split Dataset
    - fieldvars (list[str]): names of 4D predictor field variables (lat, lon, lev, time)
    - localvars (list[str]): names of (2D or 3D) local input variables
    - targetvar (str): name of 3D target variable (lat, lon, time)
    Returns:
    - tuple[torch.Tensor,torch.Tensor | None,torch.Tensor]: field, (optional) local, and target tensors
    '''
    # fields: (nfieldvars, nlats, nlons, nlevs, ntimes)
    fieldlist = []
    for varname in fieldvars:
        da  = ds[varname].transpose('lat','lon','lev','time')
        arr = da.values.astype(np.float32)
        fieldlist.append(arr)
    fielddata = torch.from_numpy(np.stack(fieldlist,axis=0))

    # locals: (nlocalvars, nlats, nlons, ntimes) or None
    if localvars:
        locallist = []
        ntimes = ds[targetvar].sizes['time']
        for varname in localvars:
            da = ds[varname]
            if 'time' in da.dims:
                arr = da.transpose('lat','lon','time').values.astype(np.float32)
            else:
                arr2d = da.transpose('lat','lon').values.astype(np.float32)
                arr   = np.broadcast_to(arr2d[...,None],
                                        (arr2d.shape[0],arr2d.shape[1],ntimes)).astype(np.float32)
            locallist.append(arr)
        localdata = torch.from_numpy(np.stack(locallist,axis=0))
    else:
        localdata = None

    # target: (nlats, nlons, ntimes)
    targetdata = torch.from_numpy(
        ds[targetvar].transpose('lat','lon','time').values.astype(np.float32))

    return fielddata,localdata,targetdata


class Patch:
    
    def __init__(self,latradius,lonradius,maxlevs,timelag):
        '''
        Purpose: Store patch configuration and infer patch shape/valid patch centers.
        Args:
        - latradius (int): number of latitude grid points to include on each side of the center
        - lonradius (int): number of longitude grid points to include on each side of the center
        - maxlevs (int): maximum number of vertical levels to include; if maxlevs ≥ total levels, use all levels
        - timelag (int): number of past time steps to include; if 0, use only the current time step (no time lag)
        '''
        self.latradius = int(latradius)
        self.lonradius = int(lonradius)
        self.maxlevs   = int(maxlevs)
        self.timelag   = int(timelag)
    
    def shape(self,nlevs):
        '''
        Purpose: Infer (plats, plons, plevs, ptimes) from patch configuration and grid size.
        Args:
        - nlevs (int): number of vertical levels in the full grid
        Returns:
        - tuple[int,int,int,int]: (plats, plons, plevs, ptimes)
        '''
        plats  = 2*self.latradius+1
        plons  = 2*self.lonradius+1
        plevs  = min(nlevs,self.maxlevs)
        ptimes = self.timelag+1 if self.timelag>0 else 1
        return (plats,plons,plevs,ptimes)
    
    def centers(self,targetdata,lats,lons,latrange,lonrange):
        '''
        Purpose: Build (latidx, lonidx, timeidx) centers where the patch fits, target is finite, and (lat, lon) lies 
        inside the prediction domain.
        Args:
        - targetdata (torch.Tensor): (nlats, nlons, ntimes)
        - lats (np.ndarray): latitude values with shape (nlats,)
        - lons (np.ndarray): longitude values with shape (nlons,)
        - latrange (tuple[float,float]): minimum/maximum latitude bounds
        - lonrange (tuple[float,float]): minimum/maximum longitude bounds
        Returns:
        - list[tuple[int,int,int]]: list of (latidx, lonidx, timeidx)
        '''
        nlats,nlons,ntimes = targetdata.shape
        lat_idx,lon_idx,time_idx = torch.nonzero(torch.isfinite(targetdata),as_tuple=True)

        # patch must fit spatially inside halo grid
        latrad,lonrad = self.latradius,self.lonradius
        fit_mask = (
            (lat_idx >= latrad) & (lat_idx < nlats-latrad) &
            (lon_idx >= lonrad) & (lon_idx < nlons-lonrad)
        )

        # restrict to prediction domain
        lat_vals = torch.as_tensor(lats,dtype=torch.float32)
        lon_vals = torch.as_tensor(lons,dtype=torch.float32)
        latmin,latmax = latrange
        lonmin,lonmax = lonrange
        dom_mask = (
            (lat_vals[lat_idx] >= latmin) & (lat_vals[lat_idx] <= latmax) &
            (lon_vals[lon_idx] >= lonmin) & (lon_vals[lon_idx] <= lonmax)
        )

        keep = fit_mask & dom_mask

        lat_sel  = lat_idx[keep].tolist()
        lon_sel  = lon_idx[keep].tolist()
        time_sel = time_idx[keep].tolist()
        return list(zip(lat_sel,lon_sel,time_sel))


class SampleDataset(Dataset):
    
    def __init__(self,fielddata,localdata,targetdata,centers,patch):
        '''
        Purpose: Return patches, optional local inputs, and target values for (lat, lon, time) samples.
        Args:
        - fielddata (torch.Tensor): (nfieldvars, nlats, nlons, nlevs, ntimes)
        - localdata (torch.Tensor | None): (nlocalvars, nlats, nlons, ntimes)
        - targetdata (torch.Tensor): (nlats, nlons, ntimes)
        - centers (list[tuple[int,int,int]]): (latidx, lonidx, timeidx) patch centers
        - patch (Patch): patch configuration
        '''
        super().__init__()
        if fielddata.ndim!=5:
            raise ValueError('`fielddata` must have shape (nfieldvars, nlats, nlons, nlevs, ntimes)')
        if localdata is not None and localdata.ndim!=4:
            raise ValueError('`localdata` must have shape (nlocalvars, nlats, nlons, ntimes) or be None')
        if targetdata.ndim!=3:
            raise ValueError('`targetdata` must have shape (nlats, nlons, ntimes)')
        self.fielddata  = fielddata
        self.localdata  = localdata 
        self.targetdata = targetdata
        self.centers    = list(centers)
        self.patch      = patch
    
    def __len__(self):
        return len(self.centers)

    def __getitem__(self,idx):
        latidx,lonidx,timeidx      = self.centers[idx]
        _,nlats,nlons,nlevs,ntimes = self.fielddata.shape

        latrad = self.patch.latradius
        lonrad = self.patch.lonradius
        maxlevs = min(nlevs,self.patch.maxlevs)
        lag     = self.patch.timelag

        latmin,latmax = latidx-latrad, latidx+latrad+1
        lonmin,lonmax = lonidx-lonrad, lonidx+lonrad+1
        levmin,levmax = 0, maxlevs

        if lag>0:
            timemin_raw = timeidx - lag
            timemin     = max(0,timemin_raw)
            timemax     = timeidx + 1
            patch_len   = timemax - timemin
            needed_len  = lag + 1
        else:
            timemin = timeidx
            timemax = timeidx + 1
            patch_len = needed_len = 1

        patch_slice = self.fielddata[:,latmin:latmax,lonmin:lonmax,levmin:levmax,timemin:timemax]

        # left-pad in time if we don't have enough history
        if patch_len < needed_len:
            pad_len = needed_len - patch_len
            pad_shape = (
                patch_slice.shape[0],
                patch_slice.shape[1],
                patch_slice.shape[2],
                patch_slice.shape[3],
                pad_len
            )
            pad   = torch.zeros(pad_shape,dtype=self.fielddata.dtype)
            patch = torch.cat([pad,patch_slice],dim=-1)
        else:
            patch = patch_slice

        sample = {
            'patch':patch,
            'target':self.targetdata[latidx,lonidx,timeidx]
        }
        if self.localdata is not None:
            sample['local'] = self.localdata[:,latidx,lonidx,timeidx]
        return sample