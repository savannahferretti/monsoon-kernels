#!/usr/bin/env python

import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset,DataLoader


class DataPrep:
    
    @staticmethod
    def load(splitname,filedir):
        '''
        Purpose: Load a split as an xr.Dataset.
        Args:
        - splitname (str): 'train' | 'valid' | 'test'
        - filedir (str): directory containing split files
        Returns:
        - xr.Dataset: loaded split Dataset
        '''
        filename = f'{splitname}.h5'
        filepath = os.path.join(filedir,filename)
        ds = xr.open_dataset(filepath,engine='h5netcdf')
        return ds

    @staticmethod
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
        fieldlist = []
        for varname in fieldvars:
            da  = ds[varname]
            arr = da.values
            fieldlist.append(arr)
        fielddata = torch.from_numpy(np.stack(fieldlist,axis=0))
        if localvars:
            locallist = []
            for varname in localvars:
                da = ds[varname]

                    arr = da.values if 'time' in da.dims else np.broadcast_to(da.values[..., None],(da.values.shape[0],da.values.shape[1],ds.time.size))
                locallist.append(arr)
            localdata = torch.from_numpy(np.stack(locallist,axis=0))
        else:
            localdata = None
        targetdata = torch.from_numpy(ds[targetvar].transpose('lat', 'lon', 'time').values.astype(np.float32))
        return fielddata, localdata, targetdata

    @staticmethod
    def extract_quadweights(ds):
        '''
        Purpose: Extract quadrature weights from a split Dataset and convert to torch.Tensor.
        Args:
        - ds (xr.Dataset): split Dataset containing 'quadweights' variable
        Returns:
        - torch.Tensor: quadrature weights with shape (plats, plons, plevs, ptimes)
        '''
        if 'quadweights' not in ds:
            raise ValueError('Dataset does not contain quadweights variable')
        quadda = ds['quadweights'].transpose('lat', 'lon', 'lev', 'time')
        quadarr = quadda.values.astype(np.float32)
        return torch.from_numpy(quadarr)

    @staticmethod
    def prepare_splits(filedir, fieldvars, localvars, targetvar, splits):
        '''
        Purpose: Load splits, convert to tensors, and extract quadweights and coordinates.
        Args:
        - filedir (str): directory containing split files
        - fieldvars (list[str]): names of 4D predictor field variables
        - localvars (list[str]): names of local input variables
        - targetvar (str): name of target variable
        - splits (list[str]): list of split names to load (e.g., ['train', 'valid'])
        Returns:
        - dict: dictionary with keys for each split containing tensors, datasets, coordinates, and quadweights
        '''
        result = {}
        for splitname in splits:
            ds = DataPrep.load(splitname, filedir)
            fielddata, localdata, targetdata = DataPrep.tensors(ds, fieldvars, localvars, targetvar)
            quadweights = DataPrep.extract_quadweights(ds)
            lats = ds.lat.values
            lons = ds.lon.values
            result[splitname] = {
                'ds': ds,
                'fielddata': fielddata,
                'localdata': localdata,
                'targetdata': targetdata,
                'quadweights': quadweights,
                'lats': lats,
                'lons': lons
            }
        return result

    @staticmethod
    def build_datasets_and_loaders(splitdata, patchcfg, uselocal, latrange, lonrange, batchsize, workers, device, nlevs):
        '''
        Purpose: Build Patch, centers, SampleDatasets, and DataLoaders for given splits.
        Args:
        - splitdata (dict): dictionary from prepare_splits() containing tensors and coordinates
        - patchcfg (dict): patch configuration from model config
        - uselocal (bool): whether to use local inputs
        - latrange (tuple[float,float]): latitude range for domain filtering
        - lonrange (tuple[float,float]): longitude range for domain filtering
        - batchsize (int): batch size for DataLoader
        - workers (int): number of DataLoader workers
        - device (str): device for training ('cuda' or 'cpu')
        - nlevs (int): number of vertical levels
        Returns:
        - dict: dictionary with 'patch', 'patchshape', 'centers', 'datasets', 'loaders', 'quadweights'
        '''
        patch = Patch(
            radius=patchcfg['radius'],
            maxlevs=patchcfg['maxlevs'],
            timelag=patchcfg['timelag'])
        patchshape = patch.shape(nlevs)
        
        commonkwargs = dict(
            num_workers=workers,
            pin_memory=(device == 'cuda'),
            persistent_workers=(workers > 0))
        if workers > 0:
            commonkwargs['prefetch_factor'] = 2
        
        centers = {}
        datasets = {}
        loaders = {}
        quadweights = None
        
        for splitname, data in splitdata.items():
            centers[splitname] = patch.centers(
                targetdata=data['targetdata'],
                lats=data['lats'],
                lons=data['lons'],
                latrange=latrange,
                lonrange=lonrange)
            
            datasets[splitname] = SampleDataset(
                fielddata=data['fielddata'],
                localdata=data['localdata'],
                targetdata=data['targetdata'],
                centers=centers[splitname],
                patch=patch,
                uselocal=uselocal)
            
            shuffle = (splitname == 'train')
            loaders[splitname] = DataLoader(
                datasets[splitname],
                batch_size=batchsize,
                shuffle=shuffle,
                **commonkwargs)
            
            if quadweights is None:
                quadweights = data['quadweights'][:patchshape[0], :patchshape[1], :patchshape[2], :patchshape[3]]
        
        return {
            'patch': patch,
            'patchshape': patchshape,
            'centers': centers,
            'datasets': datasets,
            'loaders': loaders,
            'quadweights': quadweights
        }


class Patch:
    
    def __init__(self, radius, maxlevs, timelag):
        '''
        Purpose: Store patch configuration and infer patch shape/valid patch centers.
        Args:
        - radius (int): number of horizontal grid points to include on each side of the center
        - maxlevs (int): maximum number of vertical levels to include; if maxlevs ≥ total levels, use all levels
        - timelag (int): number of past time steps to include; if 0, use only the current time step (no time lag)
        '''
        self.radius = int(radius)
        self.maxlevs = int(maxlevs)
        self.timelag = int(timelag)
    
    def shape(self, nlevs):
        '''
        Purpose: Infer (plats, plons, plevs, ptimes) from patch configuration and grid size.
        Args:
        - nlevs (int): number of vertical levels in the full grid
        Returns:
        - tuple[int,int,int,int]: (plats, plons, plevs, ptimes)
        '''
        plats = 2 * self.radius + 1
        plons = 2 * self.radius + 1
        plevs = min(nlevs, self.maxlevs)
        ptimes = self.timelag + 1 if self.timelag > 0 else 1
        return (plats, plons, plevs, ptimes)
    
    def centers(self, targetdata, lats, lons, latrange, lonrange):
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
        nlats, nlons, ntimes = targetdata.shape
        latidx, lonidx, timeidx = torch.nonzero(torch.isfinite(targetdata), as_tuple=True)
        lats = torch.as_tensor(lats, dtype=torch.float32)
        lons = torch.as_tensor(lons, dtype=torch.float32)
        fitmask = ((latidx >= self.radius) & (latidx < nlats - self.radius) &
                   (lonidx >= self.radius) & (lonidx < nlons - self.radius))
        domainmask = ((lats[latidx] >= latrange[0]) & (lats[latidx] <= latrange[1]) &
                      (lons[lonidx] >= lonrange[0]) & (lons[lonidx] <= lonrange[1]))
        keep = fitmask & domainmask
        latidxkeep = latidx[keep].tolist()
        lonidxkeep = lonidx[keep].tolist()
        timeidxkeep = timeidx[keep].tolist()
        return list(zip(latidxkeep, lonidxkeep, timeidxkeep))


class SampleDataset(Dataset):
    
    def __init__(self, fielddata, localdata, targetdata, centers, patch, uselocal):
        '''
        Purpose: Return predictor field patches, optional local inputs, and target values for (lat, lon, time) samples.
        Args:
        - fielddata (torch.Tensor): with shape (nfieldvars, nlats, nlons, nlevs, ntimes)
        - localdata (torch.Tensor | None): with shape (nlocalvars, nlats, nlons, ntimes)
        - targetdata (torch.Tensor): with shape (nlats, nlons, ntimes)
        - centers (list[tuple[int,int,int]]): (latidx, lonidx, timeidx) patch centers
        - patch (Patch): patch configuration
        - uselocal (bool): whether to include local inputs in samples
        '''
        super().__init__()
        if fielddata.ndim != 5:
            raise ValueError('`fielddata` must have shape (nfieldvars, nlats, nlons, nlevs, ntimes)')
        if localdata is not None and localdata.ndim != 4:
            raise ValueError('`localdata` must have shape (nlocalvars, nlats, nlons, ntimes) or be None')
        if targetdata.ndim != 3:
            raise ValueError('`targetdata` must have shape (nlats, nlons, ntimes)')
        self.fielddata = fielddata
        self.localdata = localdata
        self.targetdata = targetdata
        self.centers = list(centers)
        self.patch = patch
        self.uselocal = uselocal
    
    def __len__(self):
        '''
        Purpose: Return the number of samples in the dataset.
        Returns:
        - int: number of samples
        '''
        return len(self.centers)

    def __getitem__(self,idx):
        '''
        Purpose: Extract a single sample containing a predictor field patch, optional local inputs, and target value.
        Args:
        - idx (int): sample index
        Returns:
        - dict: sample with keys 'patch', 'target', and optionally 'local'
        '''
        latidx,lonidx,timeidx      = self.centers[idx]
        _,nlats,nlons,nlevs,ntimes = self.fielddata.shape
        latmin,latmax = latidx-self.patch.radius,latidx+self.patch.radius+1
        lonmin,lonmax = lonidx-self.patch.radius,lonidx+self.patch.radius+1
        levmin,levmax = 0,min(nlevs,self.patch.maxlevs)
        if self.patch.timelag>0:
            timemin,timemax = max(0,timeidx-self.patch.timelag),timeidx+1
            patchlength  = timemax-timemin
            neededlength = self.patch.timelag+1
        else:
            timemin = timeidx
            timemax = timeidx+1
            patchlength = neededlength = 1
        patchslice = self.fielddata[:,latmin:latmax,lonmin:lonmax,levmin:levmax,timemin:timemax]
        if patchlength<neededlength:
            padlength = neededlength-patchlength
            padshape  = (patchslice.shape[0],patchslice.shape[1],patchslice.shape[2],patchslice.shape[3],padlength)
            pad   = torch.zeros(padshape,dtype=self.fielddata.dtype)
            patch = torch.cat([pad,patchslice],dim=-1)
        else:
            patch = patchslice
        sample = {
            'patch':patch,
            'target':self.targetdata[latidx,lonidx,timeidx]}
        if self.uselocal and self.localdata is not None:
            sample['local'] = self.localdata[:,latidx,lonidx,timeidx]
        return sample