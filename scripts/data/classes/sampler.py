#!/usr/bin/env python

import os
import torch
import numpy as np
import xarray as xr

class PatchDataset(torch.utils.data.Dataset):

    def __init__(self,radius,maxlevs,timelag,field,darea,dlev,dtime,local,target,uselocal,lats,lons,latrange,lonrange):
        '''
        Purpose: Initialize dataset for extracting spatial-temporal patches from climate data.
        Args:
        - radius (int): number of horizontal grid points on each side of center
        - maxlevs (int): maximum number of vertical levels
        - timelag (int): number of past time steps; if 0, use only current time
        - field (torch.Tensor): predictor fields with shape (nfieldvars, nlats, nlons, nlevs, ntimes)
        - darea (torch.Tensor): horizontal area weights with shape (nlats, nlons)
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        - dtime (torch.Tensor): time step weights with shape (ntimes,)
        - local (torch.Tensor | None): local inputs with shape (nlocalvars, nlats, nlons, ntimes) or None
        - target (torch.Tensor): target values with shape (nlats, nlons, ntimes)
        - uselocal (bool): whether to use local inputs
        - lats (np.ndarray): latitude values with shape (nlats,)
        - lons (np.ndarray): longitude values with shape (nlons,)
        - latrange (tuple[float,float]): latitude range for valid patches
        - lonrange (tuple[float,float]): longitude range for valid patches
        '''
        super().__init__()
        if field.ndim!=5:
            raise ValueError('`field` must have shape (nfieldvars, nlats, nlons, nlevs, ntimes)')
        if darea.ndim!=2:
            raise ValueError('`darea` must have shape (nlats, nlons)')
        if dlev.ndim!=1:
            raise ValueError('`dlev` must have shape (nlevs,)')
        if dtime.ndim!=1:
            raise ValueError('`dtime` must have shape (ntimes,)')
        if local is not None and local.ndim!=4:
            raise ValueError('`local` must have shape (nlocalvars, nlats, nlons, ntimes) or be None')
        if target.ndim!=3:
            raise ValueError('`target` must have shape (nlats, nlons, ntimes)')
        self.radius  = int(radius)
        self.maxlevs = int(maxlevs)
        self.timelag = int(timelag)
        self.field   = field
        self.darea   = darea
        self.dlev    = dlev
        self.dtime   = dtime
        self.local   = local
        self.target  = target
        self.uselocal = bool(uselocal)
        nlats,nlons,ntimes = target.shape
        latidxs,lonidxs,timeidxs = torch.nonzero(torch.isfinite(target),as_tuple=True)
        lats = torch.as_tensor(lats,dtype=torch.float32)
        lons = torch.as_tensor(lons,dtype=torch.float32)
        patchfits = ((latidxs>=self.radius)&(latidxs<nlats-self.radius)&(lonidxs>=self.radius)&(lonidxs<nlons-self.radius))
        indomain  = ((lats[latidxs]>=latrange[0])&(lats[latidxs]<=latrange[1])&(lons[lonidxs]>=lonrange[0])&(lons_t[lonidxs]<=lonrange[1]))
        valid = patchfits&indomain
        self.centers = list(zip(latidxs[valid].tolist(),lonidxs[valid].tolist(),timeidxs[valid].tolist()))

    def __len__(self):
        '''
        Purpose: Return number of valid patch centers.
        Returns:
        - int: number of centers
        '''
        return len(self.centers)

    def __getitem__(self,idx):
        '''
        Purpose: Return patch center indices for batch extraction.
        Args:
        - idx (int): index into centers list
        Returns:
        - tuple[int,int,int]: (latidx, lonidx, timeidx) patch center
        '''
        return self.centers[idx]

    @staticmethod
    def collate(batch,dataset):
        '''
        Purpose: Vectorized batch patch extraction from dataset.
        Args:
        - batch (list[tuple[int,int,int]]): list of (latidx, lonidx, timeidx) centers
        - dataset (PatchDataset): dataset instance
        Returns:
        - dict[str,torch.Tensor]: dictionary with fieldpatch, dareapatch, dlevpatch, dtimepatch, targetvalues, and optionally localvalues
        '''
        latidx  = torch.tensor([b[0] for b in batch],dtype=torch.long)
        lonidx  = torch.tensor([b[1] for b in batch],dtype=torch.long)
        timeidx = torch.tensor([b[2] for b in batch],dtype=torch.long)
        radius   = dataset.radius
        maxlevs  = dataset.maxlevs
        timelag  = dataset.timelag
        lat_off = torch.arange(-radius,radius+1,dtype=torch.long)
        lon_off = torch.arange(-radius,radius+1,dtype=torch.long)
        lat_grid = latidx[:,None]+lat_off[None,:]
        lon_grid = lonidx[:,None]+lon_off[None,:]
        lev_idx = torch.arange(maxlevs,dtype=torch.long)
        if timelag>0:
            time_off = torch.arange(-timelag,1,dtype=torch.long)
            time_grid = timeidx[:,None]+time_off[None,:]
            tmask = time_grid<0
            time_grid_clamped = time_grid.clamp(min=0)
        else:
            time_grid_clamped = timeidx[:,None]
            tmask = None
        field = dataset.field
        nfieldvars = field.shape[0]
        plats  = lat_grid.shape[1]
        plons  = lon_grid.shape[1]
        plevs  = lev_idx.shape[0]
        ptimes = time_grid_clamped.shape[1]
        lat_ix = lat_grid[:,:,None].expand(-1,-1,plons)
        lon_ix = lon_grid[:,None,:].expand(-1,plats,-1)
        lat_ix6 = lat_ix[:,None,:,:,None,None]
        lon_ix6 = lon_ix[:,None,:,:,None,None]
        lev_ix6 = lev_idx[None,None,None,None,:,None]
        tim_ix6 = time_grid_clamped[:,None,None,None,None,:]
        fieldpatch = field[:,lat_ix6.squeeze(1),lon_ix6.squeeze(1),lev_ix6.squeeze(0).squeeze(0),tim_ix6.squeeze(1)]
        fieldpatch = fieldpatch.permute(1,0,2,3,4,5).contiguous()
        if timelag>0 and tmask is not None and tmask.any():
            tmask6 = tmask[:,None,None,None,None,:].expand(-1,nfieldvars,plats,plons,plevs,-1)
            fieldpatch = fieldpatch.masked_fill(tmask6,0)
        darea = dataset.darea
        dareapatch = darea[lat_ix,lon_ix].contiguous()
        dlevpatch = dataset.dlev[lev_idx][None,:].expand(latidx.shape[0],-1).contiguous()
        dtime = dataset.dtime
        dtimepatch = dtime[time_grid_clamped].contiguous()
        if timelag>0 and tmask is not None and tmask.any():
            dtimepatch = dtimepatch.masked_fill(tmask,0)
        targetvalues = dataset.target[latidx,lonidx,timeidx].contiguous()
        out = {
            'fieldpatch':fieldpatch,
            'dareapatch':dareapatch,
            'dlevpatch':dlevpatch,
            'dtimepatch':dtimepatch,
            'targetvalues':targetvalues}
        if dataset.uselocal and dataset.local is not None:
            localvalues = dataset.local[:,latidx,lonidx,timeidx].permute(1,0).contiguous()
            out['localvalues'] = localvalues
        return out

class PatchDataLoader:

    @staticmethod
    def prepare(splits,fieldvars,localvars,targetvar,filedir):
        '''
        Purpose: Load data splits and prepare tensors for patch extraction.
        Args:
        - splits (list[str]): list of split names
        - fieldvars (list[str]): list of field variable names
        - localvars (list[str]): list of local variable names
        - targetvar (str): target variable name
        - filedir (str): directory containing split files
        Returns:
        - dict[str,dict]: dictionary mapping split names to data dictionaries
        '''
        result = {}
        for split in splits:
            filename = f'{split}.h5'
            filepath = os.path.join(filedir,filename)
            ds = xr.open_dataset(filepath,engine='h5netcdf')
            field = torch.from_numpy(np.stack([ds[varname].values for varname in fieldvars],axis=0))
            local = (torch.from_numpy(np.stack([
                ds[varname].values if 'time' in ds[varname].dims else
                np.broadcast_to(ds[varname].values[...,np.newaxis],(*ds[varname].shape,len(ds.time)))
                for varname in localvars],axis=0))) if localvars else None
            target = torch.from_numpy(ds[targetvar].values)
            darea = torch.from_numpy(ds['darea'].values)
            dlev  = torch.from_numpy(ds['dlev'].values)
            dtime = torch.from_numpy(ds['dtime'].values)
            result[split] = {
                'refda':ds[targetvar],
                'field':field,
                'local':local,
                'target':target,
                'darea':darea,
                'dlev':dlev,
                'dtime':dtime,
                'lats':ds.lat.values,
                'lons':ds.lon.values}
        return result

    @staticmethod
    def dataloaders(splitdata,patchconfig,uselocal,latrange,lonrange,batchsize,workers,device):
        '''
        Purpose: Create PyTorch DataLoaders for all splits.
        Args:
        - splitdata (dict): dictionary from prepare method
        - patchconfig (dict): patch configuration with radius, maxlevs, timelag
        - uselocal (bool): whether to use local inputs
        - latrange (tuple[float,float]): latitude range
        - lonrange (tuple[float,float]): longitude range
        - batchsize (int): batch size for DataLoader
        - workers (int): number of worker processes
        - device (str): device type
        Returns:
        - dict: dictionary with datasets and loaders
        '''
        kwargs = dict(num_workers=workers,pin_memory=(device=='cuda'),persistent_workers=(workers>0))
        if workers>0:
            kwargs['prefetch_factor'] = 4
        datasets = {}
        loaders  = {}
        for split,data in splitdata.items():
            datasets[split] = PatchDataset(
                patchconfig['radius'],
                patchconfig['maxlevs'],
                patchconfig['timelag'],
                data['field'],
                data['darea'],
                data['dlev'],
                data['dtime'],
                data['local']
                data['target'],
                uselocal,
                data['lats'],
                data['lons'],
                latrange,
                lonrange)
            loaders[split] = torch.utils.data.DataLoader(
                datasets[split],
                batch_size=batchsize,
                shuffle=(split=='train'),
                collate_fn=lambda batch,ds=datasets[split]:PatchDataset.collate(batch,ds),**kwargs)
        return {'datasets':datasets,'loaders':loaders}