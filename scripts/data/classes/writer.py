#!/usr/bin/env python

import os
import logging
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

class PredictionWriter:

    def __init__(self,fieldvars):
        '''
        Purpose: Store metadata needed for formatting outputs.
        Args:
        - fieldvars (list[str]): predictor variable names
        '''
        self.fieldvars = list(fieldvars)

    def to_array(self,data,kind,*,centers=None,refda=None,nkernels=None,kerneldims=None,patchshape=None,nonparam=False):
        '''
        Purpose: Put raw outputs (predictions/features/weights) into the right dense ndarray shape.
        Args:
        - data (np.ndarray): raw output array
        - kind (str): predictions, features, or weights
        - centers (list[tuple[int,int,int]] | None): list of (latidx, lonidx, timeidx) patch centers (preds/features)
        - refda (xr.DataArray | None): reference DataArray with target grid (preds/features)
        - nkernels (int | None): number of kernels (preds/features)
        - kerneldims (list[str] | tuple[str] | None): dims the kernel varies along (weights/features)
        - patchshape (tuple[int,int,int,int] | None): patch shape as (plats, plons, plevs, ptimes) (features)
        - nonparam (bool): whether kernel is non-parametric
        Returns:
        - tuple[np.ndarray,dict]: (arr, meta) where meta is used by to_dataset()
        '''
        fieldvars = self.fieldvars
        if kind in ('predictions','features') and (refda is None or centers is None):
            raise ValueError('`refda` and `centers` required for prediction/feature formatting')

        if kind=='predictions':
            nlats,nlons,ntimes = refda.shape
            member = bool(nonparam and data.ndim==2 and data.shape[1]==nkernels)
            arr = np.full((nkernels,nlats,nlons,ntimes) if member else refda.shape,np.nan,dtype=np.float32)
            for i,(latidx,lonidx,timeidx) in enumerate(centers):
                if member:
                    arr[:,latidx,lonidx,timeidx] = data[i]
                else:
                    arr[latidx,lonidx,timeidx] = data[i]
            return arr,{'kind':'predictions','member':member}

        if kind=='features':
            nlats,nlons,ntimes = refda.shape

            if data.ndim < 4:
                raise ValueError('Feature array must have shape (nsamples, field, member, ...)')

            if kerneldims is None or patchshape is None:
                raise ValueError('`kerneldims` and `patchshape` required for feature formatting')

            if nkernels is None:
                raise ValueError('`nkernels` required for feature formatting')

            nsamples = data.shape[0]
            if data.shape[1] != len(fieldvars):
                raise ValueError('`data.shape[1]` must equal len(fieldvars)')
            if data.shape[2] != nkernels:
                raise ValueError('`data.shape[2]` must equal nkernels')

            kerneldims = tuple(kerneldims)
            plats,plons,plevs,ptimes = patchshape

            # preserved dims = dims NOT integrated over
            remdims  = []
            remshape = []
            if 'lat' not in kerneldims:
                remdims.append('patch_lat')
                remshape.append(plats)
            if 'lon' not in kerneldims:
                remdims.append('patch_lon')
                remshape.append(plons)
            if 'lev' not in kerneldims:
                remdims.append('patch_lev')
                remshape.append(plevs)
            if 'time' not in kerneldims:
                remdims.append('patch_time')
                remshape.append(ptimes)

            if tuple(data.shape[3:]) != tuple(remshape):
                raise ValueError(f'Preserved feature dims mismatch: got {data.shape[3:]}, expected {tuple(remshape)}')
            arr = np.full((nkernels,len(fieldvars),*remshape,nlats,nlons,ntimes),np.nan,dtype=np.float32)

            for i,(latidx,lonidx,timeidx) in enumerate(centers):
                block = data[i].transpose(1,0,*range(3,data[i].ndim))
                arr[...,latidx,lonidx,timeidx] = block

            return arr,{'kind':'features','remdims':remdims,'nonparam':nonparam}

        if kind=='weights':
            if kerneldims is None:
                raise ValueError('`kerneldims` required for weight formatting')
            kerneldims = tuple(kerneldims)
            alldims    = ['field','member','lat','lon','lev','time']

            # Only save weights for predictor fields (exclude validity_mask channel)
            # data has shape (nfieldvars, ...) where nfieldvars = len(fieldvars) + 1
            # We only want to save the first len(fieldvars) channels
            nfields_original = len(fieldvars)

            if nonparam:
                keep    = ('field','member')
                dims0   = ['field','member']
                coords0 = {'field':fieldvars,'member':np.arange(data.shape[1])}
            else:
                keep    = ('field',)
                dims0   = ['field']
                coords0 = {'field':fieldvars}
            dims1   = [dim for dim in ('lat','lon','lev','time') if dim in kerneldims]
            indexer = [slice(None) if (dim in keep or dim in kerneldims) else 0 for dim in alldims]

            # Extract only the first nfields_original channels (exclude validity_mask)
            arr = data[tuple(indexer)]
            if arr.shape[0] > nfields_original:
                arr = arr[:nfields_original]

            return arr,{'kind':'weights','dims0':dims0,'dims1':dims1,'coords0':coords0,'nonparam':nonparam}

        raise ValueError(f'Unknown kind `{kind}`')

    def to_dataset(self,arr,meta,*,refda=None,refds=None,nkernels=None,component_weights=None):
        '''
        Purpose: Wrap a dense ndarray into an xr.Dataset with dims/coords/attrs.
        Args:
        - arr (np.ndarray): shaped dense array from to_array()
        - meta (dict[str,object]): metadata returned by to_array()
        - refda (xr.DataArray | None): reference DataArray for coords (preds/features)
        - nkernels (int | None): number of kernels (preds/features)
        - component_weights (np.ndarray | None): component weights for mixture kernels [ncomponents, ...]
        Returns:
        - xr.Dataset: Dataset ready to save
        '''
        fieldvars = self.fieldvars
        kind = meta.get('kind')

        if kind=='predictions':
            if refda is None:
                raise ValueError('`refda` required for prediction dataset construction')
            if meta.get('member',False):
                da = xr.DataArray(arr,dims=('member',)+refda.dims,coords={'member':np.arange(nkernels),**refda.coords},name='pr')
            else:
                da = xr.DataArray(arr,dims=refda.dims,coords=refda.coords,name='pr')
            da.attrs = dict(long_name='Predicted precipitation rate (log1p-transformed and standardized)',units='N/A')
            return da.to_dataset()

        if kind=='features':
            if refda is None:
                raise ValueError('`refda` required for feature dataset construction')
            if nkernels is None:
                raise ValueError('`nkernels` required for feature dataset construction')

            remdims = tuple(meta.get('remdims', ()))
            ds = xr.Dataset()

            coords = {'member': np.arange(nkernels), **refda.coords}
            # arr shape: (member, field, *remshape, lat, lon, time)
            for ax, dim in enumerate(remdims, start=2):
                coords[dim] = np.arange(arr.shape[ax])

            for fieldidx, varname in enumerate(fieldvars):
                da = xr.DataArray(
                    arr[:, fieldidx, ...],
                    dims=('member',) + remdims + refda.dims,
                    coords=coords,
                    name=varname
                )
                da.attrs = dict(long_name=f'{varname} (kernel-integrated; preserved patch dims)', units='N/A')
                ds[varname] = da
            return ds

        if kind=='weights':
            dims   = meta['dims0'] + meta['dims1']

            coords = dict(meta['coords0'])
            start = len(meta['dims0'])
            for ax,dim in enumerate(meta['dims1'],start=start):
                if refds is not None:
                    if dim in refds.coords:
                        coords[dim] = refds.coords[dim].values
                        continue
                    if dim in refds.data_vars:
                        coords[dim] = refds[dim].values
                        continue
                coords[dim] = np.arange(arr.shape[ax])

            ds = xr.Dataset()
            long_name_base = 'Nonparametric kernel weights' if meta.get('nonparam',False) else 'Parametric kernel weights'

            # If component weights provided, create k1, k2, etc. variables
            if component_weights is not None:
                for i in range(component_weights.shape[0]):
                    comp_arr = component_weights[i]
                    # Extract only the first len(fieldvars) channels (same as for arr)
                    nfields_original = len(self.fieldvars)
                    if comp_arr.shape[0] > nfields_original:
                        comp_arr = comp_arr[:nfields_original]
                    da = xr.DataArray(comp_arr, dims=dims, coords=coords, name=f'k{i+1}')
                    da.attrs = dict(long_name=f'{long_name_base} (component {i+1})', units='N/A')
                    ds[f'k{i+1}'] = da
            else:
                # Single kernel: use 'k' as variable name
                da = xr.DataArray(arr, dims=dims, coords=coords, name='k')
                da.attrs = dict(long_name=long_name_base, units='N/A')
                ds['k'] = da

            return ds

        raise ValueError(f'Unknown kind `{kind}` in meta')

    @staticmethod
    def save(name,ds,kind,split,savedir,seed=None):
        '''
        Purpose: Save an xr.Dataset to NetCDF and verify by reopening.
        Args:
        - name (str): model name
        - ds (xr.Dataset): Dataset to save
        - kind (str): predictions, features, or weights
        - split (str): valid or test
        - savedir (str): output directory
        - seed (int | None): random seed used during training (if None, seed is omitted from filename)
        Returns:
        - bool: True if save successful, False otherwise
        '''
        os.makedirs(savedir,exist_ok=True)
        if seed is not None:
            filename = f'{name}_seed{seed}_{split}_{kind}.nc'
        else:
            filename = f'{name}_{split}_{kind}.nc'
        filepath = os.path.join(savedir,filename)
        logger.info(f'   Attempting to save {filename}...')
        try:
            ds.to_netcdf(filepath,engine='h5netcdf')
            xr.open_dataset(filepath,engine='h5netcdf').close()
            logger.info('      File write successful')
            return True
        except Exception:
            logger.exception('      Failed to save or verify')
            return False
