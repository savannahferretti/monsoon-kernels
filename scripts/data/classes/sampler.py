#!/usr/bin/env python

import os
import torch
import numpy as np
import xarray as xr

class PatchDataset(torch.utils.data.Dataset):

    def __init__(self,radius,levmode,timelag,field,darea,dlev,dtime,ps,lev,local,target,uselocal,lats,lons,latrange,lonrange,maxradius,maxtimelag):
        '''
        Purpose: Initialize dataset for extracting space-height-time patches from climate data.
        Args:
        - radius (int): number of horizontal grid points on each side of center for this model
        - levmode (str): vertical level extraction mode (surface or column)
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
        self.radius     = int(radius)
        self.levmode    = str(levmode)
        self.timelag    = int(timelag)
        self.field      = field
        self.darea      = darea
        self.dlev       = dlev
        self.dtime      = dtime
        self.ps         = ps
        self.lev        = lev
        self.local      = local
        self.target     = target
        self.uselocal   = bool(uselocal)
        self.maxradius  = int(maxradius)
        self.maxtimelag = int(maxtimelag)
        nlats,nlons,ntimes       = target.shape
        latidxs,lonidxs,timeidxs = torch.nonzero(torch.isfinite(target),as_tuple=True)
        lats,lons = torch.as_tensor(lats,dtype=torch.float32),torch.as_tensor(lons,dtype=torch.float32)
        patchfits = ((latidxs>=self.maxradius)&(latidxs<nlats-self.maxradius)&(lonidxs>=self.maxradius)&(lonidxs<nlons-self.maxradius)&(timeidxs>=self.maxtimelag))
        indomain  = ((lats[latidxs]>=latrange[0])&(lats[latidxs]<=latrange[1])&(lons[lonidxs]>=lonrange[0])&(lons[lonidxs]<=lonrange[1]))
        self.centers = list(zip(latidxs[patchfits&indomain].tolist(),lonidxs[patchfits&indomain].tolist(),timeidxs[patchfits&indomain].tolist()))

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
        plats  = 2*self.radius+1
        plons  = 2*self.radius+1
        plevs  = 1 if self.levmode=='surface' else self.lev.shape[0]
        ptimes = self.timelag+1 if self.timelag>0 else 1
        return (plats,plons,plevs,ptimes)

    @staticmethod
    def collate(batch,dataset):
        '''
        Purpose: Vectorized batch patch extraction from dataset with below-surface masking.
        Args:
        - batch (list[tuple[int,int,int]]): list of (latidx, lonidx, timeidx) centers
        - dataset (PatchDataset): dataset instance
        Returns:
        - dict[str,torch.Tensor]: dictionary with "patched" quadrature weights, and field, target and (optionally local) data
        '''
        lat1D  = torch.tensor([b[0] for b in batch],dtype=torch.long)
        lon1D  = torch.tensor([b[1] for b in batch],dtype=torch.long)
        time1D = torch.tensor([b[2] for b in batch],dtype=torch.long)
        lat2D  = lat1D[:,None]+torch.arange(-dataset.radius,dataset.radius+1,dtype=torch.long)[None,:]
        lon2D  = lon1D[:,None]+torch.arange(-dataset.radius,dataset.radius+1,dtype=torch.long)[None,:]
        if dataset.timelag>0:
            time2D   = time1D[:,None]+torch.arange(-dataset.timelag,1,dtype=torch.long)[None,:]
            timemask = time2D<0
            time2D   = time2D.clamp(min=0)
        else:
            time2D   = time1D[:,None]
            timemask = None
        nbatch     = lat1D.shape[0]
        nfieldvars = dataset.field.shape[0]
        plats      = lat2D.shape[1]
        plons      = lon2D.shape[1]
        plevs      = 1 if dataset.levmode=='surface' else dataset.lev.shape[0]
        ptimes     = time2D.shape[1]
        lat3D  = lat2D[:,:,None].expand(-1,-1,plons)
        lon3D  = lon2D[:,None,:].expand(-1,plats,-1)
        ps4D   = dataset.ps[lat3D[:,:,:,None],lon3D[:,:,:,None],time2D[:,None,None,:]]
        if dataset.levmode=='surface':
            ps5D    = ps4D[:,:,:,None,:]
            field6D = dataset.field[:,
                lat3D[:,:,:,None].expand(-1,-1,-1,ptimes).reshape(-1),
                lon3D[:,:,:,None].expand(-1,-1,-1,ptimes).reshape(-1),
                (dataset.lev[None,None,None,:,None]<=ps5D).to(torch.long).argmax(dim=3).reshape(-1),
                time2D[:,None,None,:].expand(-1,plats,plons,-1).reshape(-1)] \
                .reshape(nfieldvars,nbatch,plats,plons,ptimes).permute(1,0,2,3,4).unsqueeze(4)
            valid6D = torch.ones(nbatch,1,plats,plons,plevs,ptimes,dtype=torch.bool,device=dataset.field.device)
        else:
            field6D = dataset.field[:,lat3D[:,:,:,None,None],lon3D[:,:,:,None,None],
                torch.arange(plevs,dtype=torch.long,device=dataset.field.device)[None,None,:,None],
                time2D[:,None,None,None,:]].permute(1,0,2,3,4,5).contiguous()
            valid6D = dataset.lev[None,None,None,None,:,None]<=ps4D[:,None,:,:,None,:]
        if dataset.timelag>0 and timemask is not None and timemask.any():
            field6D = field6D.masked_fill(timemask[:,None,None,None,None,:].expand(-1,nfieldvars,plats,plons,plevs,-1),0)
            valid6D = valid6D.masked_fill(timemask[:,None,None,None,None,:].expand(-1,1,plats,plons,plevs,-1),0.0)
        field6D = field6D.masked_fill(~valid6D.expand(-1,nfieldvars,-1,-1,-1,-1),0.0)
        field6D = torch.cat([field6D,valid6D.float()],dim=1)
        dtime2D = dataset.dtime[time2D].contiguous()
        if dataset.timelag>0 and timemask is not None and timemask.any():
            dtime2D = dtime2D.masked_fill(timemask,0)
        result = {
            'fieldpatch':field6D,
            'dareapatch':dataset.darea[lat3D,lon3D].contiguous(),
            'dlevpatch':dataset.dlev[None,:].expand(nbatch,-1).contiguous(),
            'dtimepatch':dtime2D,
            'dlevfull':dataset.dlev,
            'targetvalues':dataset.target[lat1D,lon1D,time1D].contiguous()}
        if dataset.uselocal and dataset.local is not None:
            result['localvalues'] = dataset.local[:,lat1D,lon1D,time1D].permute(1,0).contiguous()
        return result

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
            result[split] = {
                'refda':ds[targetvar],
                'field':field,
                'local':local,
                'target':torch.from_numpy(ds[targetvar].values),
                'darea':torch.from_numpy(ds['darea'].values),
                'dlev':torch.from_numpy(ds['dlev'].values),
                'dtime':torch.from_numpy(ds['dtime'].values),
                'ps':torch.from_numpy(ds['ps'].values),
                'lev':torch.from_numpy(ds.lev.values),
                'lats':ds.lat.values,
                'lons':ds.lon.values}
        return result

    @staticmethod
    def dataloaders(splitdata,patchconfig,uselocal,latrange,lonrange,batchsize,workers,device,maxradius,maxtimelag):
        '''
        Purpose: Create PyTorch DataLoaders for all splits.
        Args:
        - splitdata (dict): dictionary from prepare method
        - patchconfig (dict): patch configuration with radius, levmode, timelag
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
        loaders  = {}
        centers  = {}
        for split,data in splitdata.items():
            datasets[split] = PatchDataset(
                patchconfig['radius'],
                patchconfig['levmode'],
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
                datasets[split],batch_size=batchsize,shuffle=(split=='train'),
                collate_fn=lambda batch,ds=datasets[split]:PatchDataset.collate(batch,ds),**kwargs)
            centers[split] = datasets[split].centers
        result = {
            'datasets':datasets,
            'loaders':loaders,
            'centers':centers,
            'geometry':next(iter(datasets.values()))}
        return result
