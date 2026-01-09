#!/usr/bin/env python

import os
import torch
import numpy as np
import xarray as xr

class PatchDataset(torch.utils.data.Dataset):

    def __init__(self,radius,maxlevs,timelag,field,darea,dlev,dtime,ps,lev,local,target,uselocal,lats,lons,latrange,lonrange,maxradius,maxtimelag):
        '''
        Purpose: Initialize dataset for extracting spatial-temporal patches from climate data.
        Args:
        - radius (int): number of horizontal grid points on each side of center for this model
        - maxlevs (int): maximum number of vertical levels
        - timelag (int): number of past time steps for this model; if 0, use only current time
        - field (torch.Tensor): predictor fields with shape (nfieldvars, nlats, nlons, nlevs, ntimes)
        - darea (torch.Tensor): horizontal area weights with shape (nlats, nlons)
        - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
        - dtime (torch.Tensor): time step weights with shape (ntimes,)
        - ps (torch.Tensor): surface pressure with shape (nlats, nlons, ntimes)
        - lev (torch.Tensor): pressure levels with shape (nlevs,)
        - local (torch.Tensor | None): local inputs with shape (nlocalvars, nlats, nlons, ntimes) or None
        - target (torch.Tensor): target values with shape (nlats, nlons, ntimes)
        - uselocal (bool): whether to use local inputs
        - lats (np.ndarray): latitude values with shape (nlats,)
        - lons (np.ndarray): longitude values with shape (nlons,)
        - latrange (tuple[float,float]): latitude range for valid patches
        - lonrange (tuple[float,float]): longitude range for valid patches
        - maxradius (int): maximum radius across all models for common domain
        - maxtimelag (int): maximum timelag across all models for common domain
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
        if ps.ndim!=3:
            raise ValueError('`ps` must have shape (nlats, nlons, ntimes)')
        if lev.ndim!=1:
            raise ValueError('`lev` must have shape (nlevs,)')
        if local is not None and local.ndim!=4:
            raise ValueError('`local` must have shape (nlocalvars, nlats, nlons, ntimes) or be None')
        if target.ndim!=3:
            raise ValueError('`target` must have shape (nlats, nlons, ntimes)')
        self.radius = int(radius)
        self.maxlevs = int(maxlevs)
        self.timelag = int(timelag)
        self.maxradius = int(maxradius)
        self.maxtimelag = int(maxtimelag)
        self.field = field
        self.darea = darea
        self.dlev = dlev
        self.dtime = dtime
        self.ps = ps
        self.lev = lev
        self.local = local
        self.target = target
        self.uselocal = bool(uselocal)
        nlats,nlons,ntimes = target.shape
        latidxs,lonidxs,timeidxs = torch.nonzero(torch.isfinite(target),as_tuple=True)
        lats = torch.as_tensor(lats,dtype=torch.float32)
        lons = torch.as_tensor(lons,dtype=torch.float32)
        patchfits = ((latidxs>=self.maxradius)&(latidxs<nlats-self.maxradius)&(lonidxs>=self.maxradius)&(lonidxs<nlons-self.maxradius)&(timeidxs>=self.maxtimelag))
        indomain = ((lats[latidxs]>=latrange[0])&(lats[latidxs]<=latrange[1])&(lons[lonidxs]>=lonrange[0])&(lons[lonidxs]<=lonrange[1]))
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

    def shape(self):
        '''
        Purpose: Return patch shape dimensions.
        Returns:
        - tuple[int,int,int,int]: (plats, plons, plevs, ptimes)
        '''
        plats = 2*self.radius + 1
        plons = 2*self.radius + 1
        plevs = self.maxlevs
        ptimes = self.timelag + 1 if self.timelag > 0 else 1
        return (plats,plons,plevs,ptimes)

    @staticmethod
    def collate(batch,dataset):
        '''
        Purpose: Vectorized batch patch extraction from dataset with below-surface masking.
        Args:
        - batch (list[tuple[int,int,int]]): list of (latidx, lonidx, timeidx) centers
        - dataset (PatchDataset): dataset instance
        Returns:
        - dict[str,torch.Tensor]: dictionary with fieldpatch, dareapatch, dlevpatch, dtimepatch, targetvalues, and optionally localvalues
        '''
        latidx = torch.tensor([b[0] for b in batch],dtype=torch.long)
        lonidx = torch.tensor([b[1] for b in batch],dtype=torch.long)
        timeidx = torch.tensor([b[2] for b in batch],dtype=torch.long)
        radius = dataset.radius
        maxlevs = dataset.maxlevs
        timelag = dataset.timelag
        latoff = torch.arange(-radius,radius+1,dtype=torch.long)
        lonoff = torch.arange(-radius,radius+1,dtype=torch.long)
        latgrid = latidx[:,None]+latoff[None,:]
        longrid = lonidx[:,None]+lonoff[None,:]
        ps = dataset.ps
        lev = dataset.lev
        pscenter = ps[latidx,lonidx,timeidx]
        nlevstotal = lev.shape[0]
        levidxlist = []
        for i in range(latidx.shape[0]):
            validmask = lev <= pscenter[i]
            validindices = torch.where(validmask)[0]
            if len(validindices) >= maxlevs:
                levidxlist.append(validindices[:maxlevs])
            else:
                levidxlist.append(validindices)
        levidx = torch.zeros(len(levidxlist),maxlevs,dtype=torch.long)
        for i,indices in enumerate(levidxlist):
            levidx[i,:len(indices)] = indices
            if len(indices) < maxlevs:
                levidx[i,len(indices):] = indices[-1] if len(indices) > 0 else 0
        if timelag>0:
            timeoff = torch.arange(-timelag,1,dtype=torch.long)
            timegrid = timeidx[:,None]+timeoff[None,:]
            tmask = timegrid<0
            timegridclamped = timegrid.clamp(min=0)
        else:
            timegridclamped = timeidx[:,None]
            tmask = None
        field = dataset.field
        nfieldvars = field.shape[0]
        plats = latgrid.shape[1]
        plons = longrid.shape[1]
        plevs = maxlevs
        ptimes = timegridclamped.shape[1]
        latix = latgrid[:,:,None].expand(-1,-1,plons)
        lonix = longrid[:,None,:].expand(-1,plats,-1)
        nbatch = latidx.shape[0]
        fieldpatch = torch.zeros(nbatch,nfieldvars,plats,plons,plevs,ptimes,dtype=field.dtype,device=field.device)
        for i in range(nbatch):
            for k in range(plevs):
                for t in range(ptimes):
                    fieldpatch[i,:,:,:,k,t] = field[:,latix[i],lonix[i],levidx[i,k],timegridclamped[i,t]]
        if timelag>0 and tmask is not None and tmask.any():
            tmask6 = tmask[:,None,None,None,None,:].expand(-1,nfieldvars,plats,plons,plevs,-1)
            fieldpatch = fieldpatch.masked_fill(tmask6,0)
        pspatch = ps[latix,lonix,timegridclamped[:,None,None,:]]
        for i in range(nbatch):
            levgridi = lev[levidx[i]][None,None,:,None]
            psppatchi = pspatch[i:i+1,:,:,None,:]
            belowsurfacei = levgridi > psppatchi
            belowsurfacei = belowsurfacei[:,None,:,:,:,:].expand(-1,nfieldvars,-1,-1,-1,-1)
            fieldpatch[i:i+1] = fieldpatch[i:i+1].masked_fill(belowsurfacei,float('nan'))
        darea = dataset.darea
        dareapatch = darea[latix,lonix].contiguous()
        dlevpatch = torch.zeros(nbatch,plevs,dtype=dataset.dlev.dtype,device=dataset.dlev.device)
        for i in range(nbatch):
            dlevpatch[i] = dataset.dlev[levidx[i]]
        dtime = dataset.dtime
        dtimepatch = dtime[timegridclamped].contiguous()
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
            dlev = torch.from_numpy(ds['dlev'].values)
            dtime = torch.from_numpy(ds['dtime'].values)
            ps = torch.from_numpy(ds['ps'].values)
            lev = torch.from_numpy(ds.lev.values)
            result[split] = {
                'refda':ds[targetvar],
                'field':field,
                'local':local,
                'target':target,
                'darea':darea,
                'dlev':dlev,
                'dtime':dtime,
                'ps':ps,
                'lev':lev,
                'lats':ds.lat.values,
                'lons':ds.lon.values}
        return result

    @staticmethod
    def dataloaders(splitdata,patchconfig,uselocal,latrange,lonrange,batchsize,workers,device,maxradius,maxtimelag):
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
        - maxradius (int): maximum radius across all models for common domain
        - maxtimelag (int): maximum timelag across all models for common domain
        Returns:
        - dict: dictionary with datasets and loaders
        '''
        kwargs = dict(num_workers=workers,pin_memory=(device=='cuda'),persistent_workers=(workers>0))
        if workers>0:
            kwargs['prefetch_factor'] = 4
        datasets = {}
        loaders = {}
        centers = {}
        for split,data in splitdata.items():
            datasets[split] = PatchDataset(
                patchconfig['radius'],
                patchconfig['maxlevs'],
                patchconfig['timelag'],
                data['field'],
                data['darea'],
                data['dlev'],
                data['dtime'],
                data['ps'],
                data['lev'],
                data['local'],
                data['target'],
                uselocal,
                data['lats'],
                data['lons'],
                latrange,
                lonrange,
                maxradius,
                maxtimelag)
            loaders[split] = torch.utils.data.DataLoader(
                datasets[split],
                batch_size=batchsize,
                shuffle=(split=='train'),
                collate_fn=lambda batch,ds=datasets[split]:PatchDataset.collate(batch,ds),**kwargs)
            centers[split] = datasets[split].centers
        geometry = next(iter(datasets.values()))
        return {'datasets':datasets,'loaders':loaders,'centers':centers,'geometry':geometry}