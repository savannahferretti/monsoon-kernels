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

    def to_array(self,data,kind,*,centers=None,refda=None,kerneldims=None,nonparam=False):
        '''
        Purpose: Format the prediction or kernel weights into the right NumPy array shape.
        Args:
        - data (np.ndarray): raw output array
        - kind (str): predictions or weights
        - centers (list[tuple[int,int,int]] | None): list of (latidx, lonidx, timeidx) patch centers or None
        - refda (xr.DataArray | None): reference DataArray with target grid or None
        - kerneldims (list[str] | tuple[str] | None): dims the kernel varies along or None
        - nonparam (bool): whether kernel is non-parametric
        Returns:
        - tuple[np.ndarray,dict]: formatted NumPy array and dictionary of metadata to be used by to_dataset()
        '''
        if kind=='predictions':
            if refda is None or centers is None:
                raise ValueError('`refda` and `centers` required for kind==`predictions`')
            arr = np.full(refda.shape,np.nan,dtype=np.float32)
            for i,(latidx,lonidx,timeidx) in enumerate(centers):
                arr[latidx,lonidx,timeidx] = data[i]
            return arr,{'kind':'predictions'}
        if kind=='weights':
            if kerneldims is None:
                raise ValueError('`kerneldims` required for kind==`weights`')
            arr = data[tuple(slice(None) if dim=='field' or dim in tuple(kerneldims) else 0 for dim in ['field','lat','lon','lev','time'])][:len(self.fieldvars)]
            meta = {
                'kind':'weights',
                'fixeddims':['field'],
                'kerneldims':tuple(kerneldims),
                'fixedcoords':{'field':self.fieldvars},
                'nonparam':nonparam}
            return arr,meta
        raise ValueError(f'Unknown kind `{kind}`')

    def to_dataset(self,arr,meta,*,refda=None,refds=None,components=None,seedaxis=False):
        '''
        Purpose: Wrap a NumPy array of predictions or weights into xr.Dataset with metadata.
        Args:
        - arr (np.ndarray): shaped dense array from to_array()
        - meta (dict[str,object]): metadata returned by to_array()
        - refda (xr.DataArray | None): reference DataArray for coords or None
        - components (np.ndarray | None): component weights for mixture kernels or None
        - seedaxis (bool): whether arr has seed dimension as last axis (defaults to False)
        Returns:
        - xr.Dataset: Dataset ready to save
        '''
        if meta.get('kind')=='predictions':
            if refda is None:
                raise ValueError('`refda` required for kind==`predictions`')
            dims = refda.dims+('seed',) if seedaxis else refda.dims
            coords = dict(refda.coords,**({'seed':np.arange(arr.shape[-1])} if seedaxis else {}))
            da = xr.DataArray(arr,dims=dims,coords=coords,name='pr')
            da.attrs = dict(long_name='Predicted precipitation rate (log1p-transformed and standardized)',units='N/A')
            return da.to_dataset()
        if meta.get('kind')=='weights':
            dims = tuple(meta['fixeddims']+list(meta['kerneldims']))
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
            longname = 'Nonparametric kernel weights' if meta.get('nonparam',False) else 'Parametric kernel weights'
            ds = xr.Dataset()
            if components is not None:
                indexer = tuple(slice(None) if dim=='field' or dim in meta['kerneldims'] else 0 for dim in ['field','lat','lon','lev','time'])
                for i in range(components.shape[0]):
                    comp = components[i][indexer][:len(self.fieldvars)]
                    ds[f'k{i+1}'] = xr.DataArray(comp,dims=dims,coords=coords)
                    ds[f'k{i+1}'].attrs = dict(long_name=f'{longname} (component {i+1})',units='N/A')
            else:
                ds['k'] = xr.DataArray(arr,dims=dims,coords=coords)
                ds['k'].attrs = dict(long_name=longname,units='N/A')
            return ds
        raise ValueError(f'Unknown kind `{meta.get("kind")}` in meta')

    @staticmethod
    def save(name,ds,kind,split,savedir):
        '''
        Purpose: Save an xr.Dataset to NetCDF and verify by reopening.
        Args:
        - name (str): model name
        - ds (xr.Dataset): Dataset to save
        - kind (str): predictions or weights
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
