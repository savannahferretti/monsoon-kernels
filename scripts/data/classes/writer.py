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

    def to_array(self,data,kind,*,centers=None,refda=None,kerneldims=None,patchshape=None,nonparam=False):
        '''
        Purpose: Put raw outputs (predictions/features/weights) into the right dense ndarray shape.
        Args:
        - data (np.ndarray): raw output array
        - kind (str): predictions, features, or weights
        - centers (list[tuple[int,int,int]] | None): list of (latidx, lonidx, timeidx) patch centers (preds/features)
        - refda (xr.DataArray | None): reference DataArray with target grid (preds/features)
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
            arr = np.full(refda.shape,np.nan,dtype=np.float32)
            for i,(latidx,lonidx,timeidx) in enumerate(centers):
                arr[latidx,lonidx,timeidx] = data[i]
            return arr,{'kind':'predictions'}
        if kind=='features':
            nlats,nlons,ntimes = refda.shape
            if data.ndim<3:
                raise ValueError('Feature array must have shape (nsamples, field, ...)')
            if kerneldims is None or patchshape is None:
                raise ValueError('`kerneldims` and `patchshape` required for feature formatting')
            if data.shape[1]!=len(fieldvars):
                raise ValueError('`data.shape[1]` must equal len(fieldvars)')
            kerneldims = tuple(kerneldims)
            plats,plons,plevs,ptimes = patchshape
            remdims,remshape = [],[]
            for dim,size in [('patch_lat',plats),('patch_lon',plons),('patch_lev',plevs),('patch_time',ptimes)]:
                if dim.split('_')[1] not in kerneldims:
                    remdims.append(dim)
                    remshape.append(size)
            if tuple(data.shape[2:])!=tuple(remshape):
                raise ValueError(f'Preserved feature dims mismatch: got {data.shape[2:]}, expected {tuple(remshape)}')
            arr = np.full((len(fieldvars),*remshape,nlats,nlons,ntimes),np.nan,dtype=np.float32)
            for i,(latidx,lonidx,timeidx) in enumerate(centers):
                arr[...,latidx,lonidx,timeidx] = data[i]
            return arr,{'kind':'features','remdims':remdims,'nonparam':nonparam}
        if kind=='weights':
            if kerneldims is None:
                raise ValueError('`kerneldims` required for weight formatting')
            kerneldims = tuple(kerneldims)
            nfields = len(fieldvars)
            arr = data[tuple(slice(None) if d=='field' or d in kerneldims else 0 for d in ['field','lat','lon','lev','time'])][:nfields]
            return arr,{
                'kind':'weights',
                'fixeddims':['field'],
                'kerneldims':[d for d in ('lat','lon','lev','time') if d in kerneldims],
                'fixedcoords':{'field':fieldvars},
                'nonparam':nonparam}
        raise ValueError(f'Unknown kind `{kind}`')

    def to_dataset(self,arr,meta,*,refda=None,refds=None,componentweights=None,seedaxis=False):
        '''
        Purpose: Wrap a dense ndarray into an xr.Dataset with dims/coords/attrs.
        Args:
        - arr (np.ndarray): shaped dense array from to_array()
        - meta (dict[str,object]): metadata returned by to_array()
        - refda (xr.DataArray | None): reference DataArray for coords (preds/features)
        - componentweights (np.ndarray | None): component weights for mixture kernels [ncomponents, ...]
        - seedaxis (bool): whether arr has seed dimension as last axis (defaults to False)
        Returns:
        - xr.Dataset: Dataset ready to save
        '''
        fieldvars = self.fieldvars
        kind = meta.get('kind')
        if kind=='predictions':
            if refda is None:
                raise ValueError('`refda` required for kind==`predictions`')
            dims = refda.dims+('seed',) if seedaxis else refda.dims
            coords = dict(refda.coords,**({'seed':np.arange(arr.shape[-1])} if seedaxis else {}))
            da = xr.DataArray(arr,dims=dims,coords=coords,name='pr',
                attrs=dict(long_name='Predicted precipitation rate (log1p-transformed and standardized)',units='N/A'))
            return da.to_dataset()
        if kind=='features':
            if refda is None:
                raise ValueError('`refda` required for kind==`features`')
            remdims = tuple(meta.get('remdims',()))
            coords = {**refda.coords,**{dim:np.arange(arr.shape[ax]) for ax,dim in enumerate(remdims,start=1)}}
            ds = xr.Dataset()
            for fieldidx,varname in enumerate(fieldvars):
                ds[varname] = xr.DataArray(arr[fieldidx,...],dims=remdims+refda.dims,coords=coords,name=varname,
                    attrs=dict(long_name=f'{varname} (kernel-integrated; preserved patch dims)',units='N/A'))
            return ds
        if kind=='weights':
            dims = tuple(meta['fixeddims']+meta['kerneldims'])
            coords = dict(meta['fixedcoords'])
            for ax,dim in enumerate(meta['kerneldims'],start=len(meta['fixeddims'])):
                if refds is not None and dim in refds.coords:
                    coords[dim] = refds.coords[dim].values
                elif refds is not None and dim in refds.data_vars:
                    coords[dim] = refds[dim].values
                else:
                    coords[dim] = np.arange(arr.shape[ax])
            if seedaxis:
                dims = dims+('seed',)
                coords['seed'] = np.arange(arr.shape[-1])
            ds = xr.Dataset()
            longname = 'Nonparametric kernel weights' if meta.get('nonparam',False) else 'Parametric kernel weights'
            if componentweights is not None:
                indexer = tuple(slice(None) if d=='field' or d in meta['kerneldims'] else 0 for d in ['field','lat','lon','lev','time'])
                nfields = len(fieldvars)
                for i in range(componentweights.shape[0]):
                    comp = componentweights[i][indexer]
                    if comp.shape[0]>nfields:
                        comp = comp[:nfields]
                    ds[f'k{i+1}'] = xr.DataArray(comp,dims=dims,coords=coords,name=f'k{i+1}',
                        attrs=dict(long_name=f'{longname} (component {i+1})',units='N/A'))
            else:
                ds['k'] = xr.DataArray(arr,dims=dims,coords=coords,name='k',
                    attrs=dict(long_name=longname,units='N/A'))
            return ds
        raise ValueError(f'Unknown kind `{kind}` in meta')

    @staticmethod
    def save(name,ds,kind,split,savedir):
        '''
        Purpose: Save an xr.Dataset to NetCDF and verify by reopening.
        Args:
        - name (str): model name
        - ds (xr.Dataset): Dataset to save
        - kind (str): predictions, features, or weights
        - split (str): valid or test
        - savedir (str): output directory
        Returns:
        - bool: True if save successful, False otherwise
        '''
        os.makedirs(savedir,exist_ok=True)
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
