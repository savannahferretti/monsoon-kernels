#!/usr/bin/env python

import os
import h5py
import logging
import warnings
import xarray as xr

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

FILEDIR     = '/global/cfs/cdirs/m4334/sferrett/monsoon-kernels/data/interim'
SAVEDIR     = '/global/cfs/cdirs/m4334/sferrett/monsoon-kernels/data/splits'
INPUTVARS   = ['sdo','t','q','rh','thetae','thetaestar','thetaeplus','levmask']
TARGETVAR   = 'pr'
TRAINRANGE  = ('2000','2014')
VALIDRANGE  = ('2015','2017')
TESTRANGE   = ('2018','2020')

def retrieve(varname,filedir=FILEDIR):
    '''
    Purpose: Lazily import a variable as an xr.DataArray and standardize the dimension order.
    Args:
    - varname (str): variable short name
    - filedir (str): directory containing the NetCDF files (defaults to FILEDIR)
    Returns:
    - xr.DataArray: DataArray with standardized dimensions
    '''
    filename = f'{varname}.nc'
    filepath = os.path.join(filedir,filename)
    da   = xr.open_dataarray(filepath,engine='h5netcdf')
    order = tuple(dim for dim in ('time','lat','lon','lev') if dim in da.dims)
    return da.transpose(*order) if order else da
            
def split(splitname,splitrange,chunksize=2208,inputvars=INPUTVARS,targetvar=TARGETVAR):
    '''
    Purpose: Assemble the inputs/target data and one shared mask for a given split into a single xr.Dataset,
    and prepare an encoding dictionary for HDF5 (compression and chunking).
    Args:
    - splitname (str): 'train' | 'valid' | 'test'
    - splitrange (tuple[str,str]): inclusive start/end years for the split
    - chunksize (int): number of time steps to include for chunking (defaults to 2,208 for 3-month chunks)
    - inputvars (list[str]): input variable names (defaults to INPUTVARS)
    - targetvar (str): target variable name (defaults to TARGETVAR)
    Returns:
    - dict[str,object]: dictionary mapping of items needed to save the data split to disk
    '''
    datavars = {}
    y = retrieve(targetvar).sel(time=slice(*splitrange))
    datavars[targetvar] = y
    levels = None
    for inputvar in inputvars:
        da = retrieve(inputvar)
        if 'time' not in da.dims:
            da = da.expand_dims(time=y.time)
        da = da.sel(time=slice(*splitrange))
        datavars[inputvar] = da
    ds = xr.Dataset(datavars)
    encoding = {}
    for varname,da in ds.data_vars.items():
        chunks = (chunksize,da.lat.size,da.lon.size,da.lev.size) if 'lev' in da.dims else (chunksize,da.lat.size,da.lon.size)
        encoding[varname] = {
            'compression':'lzf',
            'shuffle':True,
            'chunksizes':chunks,
            'dtype':da.dtype}
    plan = {
        'split':splitname,
        'years':splitrange,
        'ds':ds,
        'encoding':encoding}
    logger.info(f'   Compiled {splitname} split')
    return plan

def save(plan,savedir=SAVEDIR):
    '''
    Purpose: Save a compiled xr.Dataset to a HDF5 file, then verify the write by reopening.
    Args:
    - plan (dict): output of split()
    - savedir (str): output directory (defaults to SAVEDIR)
    Returns:
    - bool: True if writing and verification succeed, otherwise False
    '''
    os.makedirs(savedir,exist_ok=True)
    filename = f'{plan["split"]}.h5'
    filepath = os.path.join(savedir,filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        plan['ds'].to_netcdf(filepath,engine='h5netcdf',encoding=plan['encoding'])
        with h5py.File(filepath,'r') as _:
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

if __name__=='__main__':
    logger.info('Creating split plans...')
    planlist = [
        split('train',TRAINRANGE),
        split('valid',VALIDRANGE),
        split('test',TESTRANGE)]
    logger.info('Saving data splits...')
    for plan in planlist:
        save(plan)
        del plan