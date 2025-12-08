#!/usr/bin/env python

import os
import json
import time
import logging
import warnings

import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader

from utils import Config
from data import load,tensors,Patch,SampleDataset
from models import build_model_from_config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# -----------------------------------------------------------------------------#
# Configs and device
# -----------------------------------------------------------------------------#

CFG        = Config()
FILEDIR    = CFG.filedir
MODELDIR   = CFG.modeldir
RESULTSDIR = CFG.resultsdir

FIELDVARS  = CFG.fieldvars
LOCALVARS  = CFG.localvars
TARGETVAR  = CFG.targetvar
LATRANGE   = CFG.latrange
LONRANGE   = CFG.lonrange
PATCH_CFG  = CFG.patch
MODEL_CFGS = CFG.models

SEED       = CFG.seed
WORKERS    = CFG.workers
BATCHSIZE  = CFG.batchsize

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP_ENABLED = (DEVICE=='cuda')
if DEVICE=='cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

torch.manual_seed(SEED)
np.random.seed(SEED)

# Load precip normalization stats once
STATS_PATH = os.path.join(FILEDIR,'stats.json')
with open(STATS_PATH,'r',encoding='utf-8') as _f:
    STATS = json.load(_f)


# -----------------------------------------------------------------------------#
# Model + prediction helpers
# -----------------------------------------------------------------------------#

def load_checkpoint(model,runname,modeldir=MODELDIR):
    '''
    Purpose: Load best checkpoint for a given run name.
    '''
    filename = f'{runname}_best.pth'
    filepath = os.path.join(modeldir,filename)
    if not os.path.exists(filepath):
        logger.error(f'Checkpoint not found: {filepath}')
        return False
    state = torch.load(filepath,map_location='cpu')
    model.load_state_dict(state)
    logger.info(f'   Loaded checkpoint from {filename}')
    return True


def denormalize_precip(ynorm_flat,targetvar=TARGETVAR,stats=STATS):
    '''
    Purpose: Undo z-score + log1p normalization using pre-loaded stats.
    Args:
    - ynorm_flat (np.ndarray): normalized predictions (nsamples,)
    - targetvar (str): target variable name used to construct keys in stats
    - stats (dict): loaded statistics from stats.json
    '''
    mean = float(stats[f'{targetvar}_mean'])
    std  = float(stats[f'{targetvar}_std'])
    ylog = ynorm_flat*std+mean
    y    = np.expm1(ylog)
    return y


def predict_precip(model,loader,centers,targettemplate,device=DEVICE):
    '''
    Purpose: Run model, denormalize, and reconstruct full (lat,lon,time) field.
    '''
    model = model.to(device)
    model.eval()

    preds = []
    start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            for batch in loader:
                patch = batch['patch'].to(device,non_blocking=True)
                local = batch.get('local',None)
                if local is not None:
                    local = local.to(device,non_blocking=True)
                ypred = model(patch,local)
                preds.append(ypred.squeeze(-1).cpu().numpy())
    duration = time.time()-start
    logger.info(f'   Inference time: {duration:.1f} s')

    ynorm = np.concatenate(preds,axis=0)          # (nsamples,)
    yphys = denormalize_precip(ynorm)             # (nsamples,)

    arr = np.full(targettemplate.shape,np.nan,dtype=np.float64)
    for idx,(ilat,ilon,itime) in enumerate(centers):
        arr[ilat,ilon,itime] = yphys[idx]
    return arr


def save_precip(arr,template_da,runname,splitname,resultsdir=RESULTSDIR):
    '''
    Purpose: Save predicted precipitation to NetCDF on the target grid.
    '''
    os.makedirs(resultsdir,exist_ok=True)
    da = xr.DataArray(
        arr,
        dims=template_da.dims,
        coords=template_da.coords,
        name='pr')
    da.attrs = dict(long_name='NN-predicted precipitation rate',units='mm/hr')

    filename = f'nn_{runname}_{splitname}_pr.nc'
    filepath = os.path.join(resultsdir,filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        da.to_netcdf(filepath,engine='h5netcdf')
        with xr.open_dataset(filepath,engine='h5netcdf') as _:
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate NN precipitation models.')
    parser.add_argument(
        '--models',
        type=str,
        default='all',
        help='Comma-separated list of model names to evaluate (as in configs.json "models.name"), or "all".')
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['valid','test'],
        help='Which split to evaluate.')
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    requested = [m.strip() for m in args.models.split(',')] if args.models!='all' else None
    splitname = args.split

    logger.info(f'Evaluating models on {splitname} split...')
    dssplit = load(splitname,FILEDIR)

    logger.info('Converting split to torch tensors...')
    fieldsplit,localsplit,targetsplit = tensors(dssplit,FIELDVARS,LOCALVARS,TARGETVAR)
    target_da = dssplit[TARGETVAR].transpose('lat','lon','time')

    logger.info('Configuring patch and centers...')
    patch = Patch(
        latradius=PATCH_CFG['latradius'],
        lonradius=PATCH_CFG['lonradius'],
        maxlevs=PATCH_CFG['maxlevs'],
        timelag=PATCH_CFG['timelag'])
    nlevs_full = fieldsplit.shape[3]
    patchshape = patch.shape(nlevs_full)

    centers = patch.centers(
        targetdata=targetsplit,
        lats=dssplit['lat'].values,
        lons=dssplit['lon'].values,
        latrange=LATRANGE,
        lonrange=LONRANGE)
    logger.info(f'{splitname} samples: {len(centers)}')

    logger.info('Building SampleDataset and DataLoader...')
    dataset = SampleDataset(
        fielddata=fieldsplit,
        localdata=localsplit,
        targetdata=targetsplit,
        centers=centers,
        patch=patch)

    common_loader_kwargs = dict(
        num_workers=WORKERS,
        pin_memory=(DEVICE=='cuda'),
        persistent_workers=(WORKERS>0))
    if WORKERS>0:
        common_loader_kwargs['prefetch_factor'] = 2

    loader = DataLoader(
        dataset,batch_size=BATCHSIZE,shuffle=False,
        **common_loader_kwargs)

    nfieldvars = len(FIELDVARS)
    nlocalvars = len(LOCALVARS)

    for mcfg in MODEL_CFGS:
        name = mcfg['name']
        if (requested is not None) and (name not in requested):
            continue

        logger.info(f'=== Evaluating model: {name} ({mcfg["type"]}) on {splitname} ===')
        model = build_model_from_config(mcfg,patchshape,nfieldvars,nlocalvars)
        if not load_checkpoint(model,name):
            logger.error(f'Skipping model {name} (no checkpoint).')
            continue

        yarr = predict_precip(model,loader,centers,targetsplit,device=DEVICE)
        save_precip(yarr,target_da,name,splitname)

    logger.info('Finished evaluating selected models.')