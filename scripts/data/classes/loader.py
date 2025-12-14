# #!/usr/bin/env python

import os
import torch
import numpy as np
import xarray as xr

class PatchGeometry:

    def __init__(self, radius, maxlevs, timelag):
        self.radius  = int(radius)
        self.maxlevs = int(maxlevs)
        self.timelag = int(timelag)

    def shape(self):
        plats  = 2 * self.radius + 1
        plons  = 2 * self.radius + 1
        plevs  = self.maxlevs
        ptimes = self.timelag + 1 if self.timelag > 0 else 1
        return (plats, plons, plevs, ptimes)

    def centers(self, target, lats, lons, latrange, lonrange):
        nlats, nlons, ntimes       = target.shape
        latidxs, lonidxs, timeidxs = torch.nonzero(torch.isfinite(target), as_tuple=True)

        lats_t = torch.as_tensor(lats, dtype=torch.float32)
        lons_t = torch.as_tensor(lons, dtype=torch.float32)

        patchfits = (
            (latidxs >= self.radius) & (latidxs < nlats - self.radius) &
            (lonidxs >= self.radius) & (lonidxs < nlons - self.radius)
        )
        indomain = (
            (lats_t[latidxs] >= latrange[0]) & (lats_t[latidxs] <= latrange[1]) &
            (lons_t[lonidxs] >= lonrange[0]) & (lons_t[lonidxs] <= lonrange[1])
        )

        valid = patchfits & indomain
        return list(zip(latidxs[valid].tolist(), lonidxs[valid].tolist(), timeidxs[valid].tolist()))

class PatchDataset(torch.utils.data.Dataset):

    def __init__(self, geometry, centers, field, darea, dlev, dtime, local, target, uselocal):
        super().__init__()

        if field.ndim != 5:
            raise ValueError('`field` must have shape (nfieldvars, nlats, nlons, nlevs, ntimes)')
        if darea.ndim != 2:
            raise ValueError('`darea` must have shape (nlats, nlons)')
        if dlev.ndim != 1:
            raise ValueError('`dlev` must have shape (nlevs,)')
        if dtime.ndim != 1:
            raise ValueError('`dtime` must have shape (ntimes,)')
        if local is not None and local.ndim != 4:
            raise ValueError('`local` must have shape (nlocalvars, nlats, nlons, ntimes) or be None')
        if target.ndim != 3:
            raise ValueError('`target` must have shape (nlats, nlons, ntimes)')

        self.geometry = geometry
        self.centers  = list(centers)

        # store as tensors (CPU). DataLoader + pin_memory will handle fast H2D copies.
        self.field    = field
        self.darea    = darea
        self.dlev     = dlev
        self.dtime    = dtime
        self.local    = local
        self.target   = target
        self.uselocal = bool(uselocal)

    def __len__(self):
        return len(self.centers)

    # NEW: return only indices. Batch extraction happens in collate_fn.
    def __getitem__(self, idx):
        return self.centers[idx]  # (latidx, lonidx, timeidx)


def _collate_patches(batch, dataset: PatchDataset):
    """
    Vectorized batch patch extraction.

    Returns dict with keys:
      - fieldpatch   : (B, nfieldvars, plats, plons, plevs, ptimes)
      - dareapatch   : (B, plats, plons)
      - dlevpatch    : (B, plevs)
      - dtimepatch   : (B, ptimes)
      - targetvalues : (B,)
      - localvalues  : (B, nlocalvars)  [if enabled]
    """
    # batch is list[(latidx, lonidx, timeidx)]
    latidx = torch.tensor([b[0] for b in batch], dtype=torch.long)
    lonidx = torch.tensor([b[1] for b in batch], dtype=torch.long)
    timeidx = torch.tensor([b[2] for b in batch], dtype=torch.long)

    g = dataset.geometry
    r = g.radius
    maxlevs = g.maxlevs
    timelag = g.timelag

    # Offsets for lat/lon patch windows
    lat_off = torch.arange(-r, r + 1, dtype=torch.long)
    lon_off = torch.arange(-r, r + 1, dtype=torch.long)

    lat_grid = latidx[:, None] + lat_off[None, :]         # (B, plats)
    lon_grid = lonidx[:, None] + lon_off[None, :]         # (B, plons)

    # Levels
    lev_idx = torch.arange(maxlevs, dtype=torch.long)     # (plevs,)

    # Times: handle timelag (left-pad with zeros if insufficient history)
    if timelag > 0:
        # relative offsets: [-timelag, ..., 0]
        t_off = torch.arange(-timelag, 1, dtype=torch.long)       # (ptimes,)
        time_grid = timeidx[:, None] + t_off[None, :]             # (B, ptimes)

        # mask for negative times (need padding)
        tmask = time_grid < 0
        time_grid_clamped = time_grid.clamp(min=0)
    else:
        time_grid_clamped = timeidx[:, None]  # (B, 1)
        tmask = None

    # --- Extract patches (vectorized advanced indexing) ---
    # field: (V, LAT, LON, LEV, TIME)
    # want:  (B, V, plats, plons, plevs, ptimes)
    field = dataset.field

    V = field.shape[0]
    plats = lat_grid.shape[1]
    plons = lon_grid.shape[1]
    plevs = lev_idx.shape[0]
    ptimes = time_grid_clamped.shape[1]

    # Build broadcasted index tensors
    # We'll create indices with shapes that broadcast to (B, V, plats, plons, plevs, ptimes)
    b_idx = torch.arange(latidx.shape[0], dtype=torch.long)[:, None, None, None, None]  # (B,1,1,1,1)
    # Not used directly in indexing, but helps conceptualization.

    # Expand lat/lon to (B, plats, plons)
    lat_ix = lat_grid[:, :, None].expand(-1, -1, plons)  # (B, plats, plons)
    lon_ix = lon_grid[:, None, :].expand(-1, plats, -1)  # (B, plats, plons)

    # Expand to full 6D indexing
    # We index field with:
    #   field[:, lat_ix, lon_ix, lev_idx, time_grid]
    # but need to align dims. We'll add singleton dims and expand.
    lat_ix6 = lat_ix[:, None, :, :, None, None]  # (B,1,plats,plons,1,1)
    lon_ix6 = lon_ix[:, None, :, :, None, None]  # (B,1,plats,plons,1,1)
    lev_ix6 = lev_idx[None, None, None, None, :, None]   # (1,1,1,1,plevs,1)
    tim_ix6 = time_grid_clamped[:, None, None, None, None, :]  # (B,1,1,1,1,ptimes)

    # Add V dimension by expanding after selecting V with slice
    # Result from advanced indexing will be (V, B, plats, plons, plevs, ptimes) if we include V as first dim.
    fieldpatch = field[:, lat_ix6.squeeze(1), lon_ix6.squeeze(1), lev_ix6.squeeze(0).squeeze(0), tim_ix6.squeeze(1)]
    # fieldpatch shape: (V, B, plats, plons, plevs, ptimes)
    fieldpatch = fieldpatch.permute(1, 0, 2, 3, 4, 5).contiguous()  # (B, V, plats, plons, plevs, ptimes)

    # Zero-pad fieldpatch where time_grid was negative (left padding)
    if timelag > 0 and tmask is not None and tmask.any():
        # tmask: (B, ptimes) -> expand to (B,1,1,1,1,ptimes)
        tmask6 = tmask[:, None, None, None, None, :].expand(-1, V, plats, plons, plevs, -1)
        fieldpatch = fieldpatch.masked_fill(tmask6, 0)

    # darea: (LAT, LON) -> dareapatch: (B, plats, plons)
    darea = dataset.darea
    dareapatch = darea[lat_ix, lon_ix].contiguous()

    # dlev: (LEV,) -> dlevpatch: (B, plevs)
    dlevpatch = dataset.dlev[lev_idx][None, :].expand(latidx.shape[0], -1).contiguous()

    # dtime: (TIME,) -> dtimepatch: (B, ptimes)
    dtime = dataset.dtime
    dtimepatch = dtime[time_grid_clamped].contiguous()
    if timelag > 0 and tmask is not None and tmask.any():
        dtimepatch = dtimepatch.masked_fill(tmask, 0)

    # targetvalues: (B,)
    targetvalues = dataset.target[latidx, lonidx, timeidx].contiguous()

    out = {
        'fieldpatch': fieldpatch,
        'dareapatch': dareapatch,
        'dlevpatch': dlevpatch,
        'dtimepatch': dtimepatch,
        'targetvalues': targetvalues
    }

    if dataset.uselocal and dataset.local is not None:
        # local: (L, LAT, LON, TIME) -> (B, L)
        localvalues = dataset.local[:, latidx, lonidx, timeidx].permute(1, 0).contiguous()
        out['localvalues'] = localvalues

    return out


class PatchDataLoader:

    @staticmethod
    def prepare(splits, fieldvars, localvars, targetvar, filedir):
        result = {}
        for split in splits:
            filename = f'{split}.h5'
            filepath = os.path.join(filedir, filename)
            ds = xr.open_dataset(filepath, engine='h5netcdf')

            field = torch.from_numpy(np.stack([ds[varname].values for varname in fieldvars], axis=0))

            local = (torch.from_numpy(np.stack([
                ds[varname].values if 'time' in ds[varname].dims else
                np.broadcast_to(ds[varname].values[..., np.newaxis], (*ds[varname].shape, len(ds.time)))
                for varname in localvars
            ], axis=0))) if localvars else None

            target = torch.from_numpy(ds[targetvar].values)

            darea = torch.from_numpy(ds['darea'].values)
            dlev  = torch.from_numpy(ds['dlev'].values)
            dtime = torch.from_numpy(ds['dtime'].values)

            result[split] = {
                'refda': ds[targetvar],
                'field': field,
                'local': local,
                'target': target,
                'darea': darea,
                'dlev': dlev,
                'dtime': dtime,
                'lats': ds.lat.values,
                'lons': ds.lon.values
            }
        return result

    @staticmethod
    def dataloaders(splitdata, patchconfig, uselocal, latrange, lonrange, batchsize, workers, device):
        geometry = PatchGeometry(patchconfig['radius'], patchconfig['maxlevs'], patchconfig['timelag'])

        kwargs = dict(
            num_workers=workers,
            pin_memory=(device == 'cuda'),
            persistent_workers=(workers > 0)
        )
        if workers > 0:
            kwargs['prefetch_factor'] = 4

        centers  = {}
        datasets = {}
        loaders  = {}

        for split, data in splitdata.items():
            centers[split] = geometry.centers(data['target'], data['lats'], data['lons'], latrange, lonrange)

            datasets[split] = PatchDataset(
                geometry, centers[split], data['field'],
                data['darea'], data['dlev'], data['dtime'],
                data['local'], data['target'], uselocal
            )

            loaders[split] = torch.utils.data.DataLoader(
                datasets[split],
                batch_size=batchsize,
                shuffle=(split == 'train'),
                collate_fn=lambda batch, ds=datasets[split]: _collate_patches(batch, ds),
                **kwargs
            )

        return {
            'geometry': geometry,
            'centers': centers,
            'datasets': datasets,
            'loaders': loaders
        }





# import os
# import torch
# import numpy as np
# import xarray as xr

# class PatchGeometry:

#     def __init__(self,radius,maxlevs,timelag):
#         '''
#         Purpose: Initialize patch geometry to infer patch shape and valid patch centers.
#         Args:
#         - radius (int): number of horizontal grid points to include on each side of the center point
#         - maxlevs (int): maximum number of vertical levels to include; should be ≤ total levels
#         - timelag (int): number of past time steps to include; if 0, use only the current time step (no time lag)
#         '''
#         self.radius  = int(radius)
#         self.maxlevs = int(maxlevs)
#         self.timelag = int(timelag)

#     def shape(self):
#         '''
#         Purpose: Infer the number of patch latitudes, longitudes, vertical levels, and time steps from patch
#         geometry and grid size.
#         Returns:
#         - tuple[int,int,int,int]: (plats, plons, plevs, ptimes)
#         '''
#         plats  = 2*self.radius+1
#         plons  = 2*self.radius+1
#         plevs  = self.maxlevs
#         ptimes = self.timelag+1 if self.timelag>0 else 1
#         return (plats,plons,plevs,ptimes)

#     def centers(self,target,lats,lons,latrange,lonrange):
#         '''
#         Purpose: Build (latidx, lonidx, timeidx) centers where the patch fits, the target is finite, and the
#         center lies inside the prediction domain.
#         Args:
#         - target (torch.Tensor): target values with shape (nlats, nlons, ntimes)
#         - lats (np.ndarray): latitude values with shape (nlats,)
#         - lons (np.ndarray): longitude values with shape (nlons,)
#         - latrange (tuple[float,float]): latitude range
#         - lonrange (tuple[float,float]): longitude range
#         Returns:
#         - list[tuple[int,int,int]]: list of (latidx, lonidx, timeidx) patch centers
#         '''
#         nlats,nlons,ntimes       = target.shape
#         latidxs,lonidxs,timeidxs = torch.nonzero(torch.isfinite(target),as_tuple=True)
#         lats,lons = torch.as_tensor(lats,dtype=torch.float32),torch.as_tensor(lons,dtype=torch.float32)
#         patchfits = ((latidxs>=self.radius)&(latidxs<nlats-self.radius)&(lonidxs>=self.radius)&(lonidxs<nlons-self.radius))
#         indomain  = ((lats[latidxs]>=latrange[0])&(lats[latidxs]<=latrange[1])&(lons[lonidxs]>=lonrange[0])&(lons[lonidxs]<=lonrange[1]))
#         validlatidxs  = latidxs[patchfits&indomain].tolist()
#         validlonidxs  = lonidxs[patchfits&indomain].tolist()
#         validtimeidxs = timeidxs[patchfits&indomain].tolist()
#         return list(zip(validlatidxs,validlonidxs,validtimeidxs))

# class PatchDataset(torch.utils.data.Dataset):

#     def __init__(self,geometry,centers,field,darea,dlev,dtime,local,target,uselocal):
#         '''
#         Purpose: Return patches of the predictors fields and quadrature weights (separable components + product measure),
#         as well as local (optional) and target values for each patch center.

#         Notes:
#         - The separable components (ΔA, Δp, Δt) are carried explicitly so kernel normalization can use only the factors
#           along the kernel's active dimensions.
#         - The product measure (quadpatch = ΔA*Δp*Δt) is provided for the feature integral itself.

#         Args:
#         - geometry (PatchGeometry): patch geometry
#         - centers (list[tuple[int,int,int]]): list of (latidx, lonidx, timeidx) patch centers
#         - field (torch.Tensor): predictor fields with shape (nfieldvars, nlats, nlons, nlevs, ntimes)
#         - darea (torch.Tensor): horizontal area weights with shape (nlats, nlons)
#         - dlev (torch.Tensor): vertical thickness weights with shape (nlevs,)
#         - dtime (torch.Tensor): time step weights with shape (ntimes,)
#         - local (torch.Tensor | None): local inputs with shape (nlocalvars, nlats, nlons, ntimes) if uselocal is True, otherwise None
#         - target (torch.Tensor): target values with shape (nlats, nlons, ntimes)
#         - uselocal (bool): whether to use local inputs
#         '''
#         super().__init__()
#         if field.ndim!=5:
#             raise ValueError('`field` must have shape (nfieldvars, nlats, nlons, nlevs, ntimes)')
#         if darea.ndim!=2:
#             raise ValueError('`darea` must have shape (nlats, nlons)')
#         if dlev.ndim!=1:
#             raise ValueError('`dlev` must have shape (nlevs,)')
#         if dtime.ndim!=1:
#             raise ValueError('`dtime` must have shape (ntimes,)')
#         if local is not None and local.ndim!=4:
#             raise ValueError('`local` must have shape (nlocalvars, nlats, nlons, ntimes) or be None')
#         if target.ndim!=3:
#             raise ValueError('`target` must have shape (nlats, nlons, ntimes)')
#         self.geometry = geometry
#         self.centers  = list(centers)
#         self.field    = field
#         self.darea    = darea
#         self.dlev     = dlev
#         self.dtime    = dtime
#         self.local    = local
#         self.target   = target
#         self.uselocal = bool(uselocal)

#     def __len__(self):
#         '''
#         Purpose: Return the number of valid patch centers in the dataset.
#         Returns:
#         - int: number of centers
#         '''
#         return len(self.centers)

#     @staticmethod
#     def _pad_left_1d(vec,padlength):
#         '''
#         Purpose: Left-pad a 1D tensor with zeros (used for time-lag boundary samples).
#         Args:
#         - vec (torch.Tensor): 1D tensor of shape (n,)
#         - padlength (int): number of zeros to prepend
#         Returns:
#         - torch.Tensor: padded 1D tensor of shape (n+padlength,)
#         '''
#         if padlength<=0:
#             return vec
#         pad = torch.zeros((padlength,),dtype=vec.dtype,device=vec.device)
#         return torch.cat([pad,vec],dim=0)

#     def __getitem__(self,idx):
#         '''
#         Purpose: Extract a single sample containing patches for the predictor fields and quadrature weights, as well as
#         local (optional) and target values.

#         Returns a dictionary with:
#         - fieldpatch: predictor fields patch (nfieldvars, plats, plons, plevs, ptimes)
#         - quadpatch:  product measure patch (plats, plons, plevs, ptimes) = ΔA*Δp*Δt
#         - dareapatch: horizontal area patch (plats, plons)
#         - dlevpatch:  vertical thickness patch (plevs,)
#         - dtimepatch: time step patch (ptimes,)
#         - targetvalues: scalar target value at the patch center
#         - localvalues (optional): local inputs at the patch center

#         Args:
#         - idx (int): index into valid centers list
#         Returns:
#         - dict[str,torch.Tensor]: sample dictionary
#         '''
#         latidx,lonidx,timeidx = self.centers[idx]
#         latmin,latmax = latidx-self.geometry.radius,latidx+self.geometry.radius+1
#         lonmin,lonmax = lonidx-self.geometry.radius,lonidx+self.geometry.radius+1
#         levmin,levmax = 0,self.geometry.maxlevs
#         if self.geometry.timelag>0:
#             timemin,timemax = max(0,timeidx-self.geometry.timelag),timeidx+1
#             patchtimelength = timemax-timemin
#             neededlength    = self.geometry.timelag+1
#         else:
#             timemin,timemax = timeidx,timeidx+1
#             patchtimelength = neededlength = 1
#         fieldpatch = self.field[:,latmin:latmax,lonmin:lonmax,levmin:levmax,timemin:timemax]
#         dareapatch = self.darea[latmin:latmax,lonmin:lonmax]
#         dlevpatch  = self.dlev[levmin:levmax]
#         dtimepatch = self.dtime[timemin:timemax]
#         if patchtimelength<neededlength:
#             padlength  = neededlength-patchtimelength
#             fieldpad   = torch.zeros((*fieldpatch.shape[:-1],padlength),dtype=fieldpatch.dtype,device=fieldpatch.device)
#             fieldpatch = torch.cat([fieldpad,fieldpatch],dim=-1)
#             dtimepatch = self._pad_left_1d(dtimepatch,padlength)
#         sample = {
#             'fieldpatch':fieldpatch,
#             'dareapatch':dareapatch,
#             'dlevpatch':dlevpatch,
#             'dtimepatch':dtimepatch,
#             'targetvalues':self.target[latidx,lonidx,timeidx]}
#         if self.uselocal and self.local is not None:
#             sample['localvalues'] = self.local[:,latidx,lonidx,timeidx]
#         return sample

# class PatchDataLoader:

#     @staticmethod
#     def prepare(splits,fieldvars,localvars,targetvar,filedir):
#         '''
#         Purpose: Convert variable xr.DataArrays into PyTorch tensors by data type and extract separable quadrature weights
#         and coordinates.

#         Notes:
#         - Quadrature weights are stored and propagated as separable components:
#           - darea (lat, lon)
#           - dlev  (lev,)
#           - dtime (time,)
#         - The product measure (quadpatch = darea*dlev*dtime) is constructed later at patch time (PatchDataset).

#         Args:
#         - splits (list[str]): list of data splits to load
#         - fieldvars (list[str]): predictor field variable names
#         - localvars (list[str]): local input variable names
#         - targetvar (str): target variable name
#         - filedir (str): directory containing split files
#         Returns:
#         - dict[str,dict]: dictionary mapping split names to data dictionaries containing PyTorch tensors and coordinates
#         '''
#         result = {}
#         for split in splits:
#             filename = f'{split}.h5'
#             filepath = os.path.join(filedir,filename)
#             ds = xr.open_dataset(filepath,engine='h5netcdf')
#             field = torch.from_numpy(np.stack([ds[varname].values for varname in fieldvars],axis=0))
#             local = (torch.from_numpy(np.stack([ds[varname].values if 'time' in ds[varname].dims else
#                                                  np.broadcast_to(ds[varname].values[...,np.newaxis],(*ds[varname].shape,len(ds.time)))
#                 for varname in localvars],axis=0))) if localvars else None
#             target = torch.from_numpy(ds[targetvar].values)
#             darea = torch.from_numpy(ds['darea'].values)
#             dlev  = torch.from_numpy(ds['dlev'].values)
#             dtime = torch.from_numpy(ds['dtime'].values)
#             result[split] = {
#                 'refda':ds[targetvar],
#                 'field':field,
#                 'local':local,
#                 'target':target,
#                 'darea':darea,
#                 'dlev':dlev,
#                 'dtime':dtime,
#                 'lats':ds.lat.values,
#                 'lons':ds.lon.values}
#         return result

#     @staticmethod
#     def dataloaders(splitdata,patchconfig,uselocal,latrange,lonrange,batchsize,workers,device):
#         '''
#         Purpose: Build PatchGeometry, centers, PatchDatasets, and DataLoaders for given splits.

#         Notes:
#         - PatchDataset constructs and returns:
#           - quadpatch  = darea*dlev*dtime over each sample's patch
#           - dareapatch, dlevpatch, dtimepatch (separable components for kernel normalization)
#         - Optimizations: pin_memory, persistent_workers, and prefetch_factor for GPU efficiency

#         Args:
#         - splitdata (dict): dictionary from prepare()
#         - patchconfig (dict): patch configuration
#         - uselocal (bool): whether to use local inputs
#         - latrange (tuple[float,float]): latitude range
#         - lonrange (tuple[float,float]): longitude range
#         - batchsize (int): batch size for PyTorch DataLoader
#         - workers (int): number of PyTorch DataLoader workers
#         - device (str): device to use
#         Returns:
#         - dict[str,object]: dictionary containing the patch geometry, valid patch centers, constructed datasets,
#           and dataloaders
#         '''
#         geometry = PatchGeometry(patchconfig['radius'],patchconfig['maxlevs'],patchconfig['timelag'])
#         kwargs   = dict(num_workers=workers,pin_memory=(device=='cuda'),persistent_workers=(workers>0))
#         if workers>0:
#             kwargs['prefetch_factor'] = 4
#         centers  = {}
#         datasets = {}
#         loaders  = {}
#         for split,data in splitdata.items():
#             centers[split]  = geometry.centers(data['target'],data['lats'],data['lons'],latrange,lonrange)
#             datasets[split] = PatchDataset(geometry,centers[split],data['field'],
#                                            data['darea'],data['dlev'],data['dtime'],
#                                            data['local'],data['target'],uselocal)
#             loaders[split]  = torch.utils.data.DataLoader(datasets[split],batch_size=batchsize,shuffle=(split=='train'),**kwargs)
#         return {
#             'geometry':geometry,
#             'centers':centers,
#             'datasets':datasets,
#             'loaders':loaders}
