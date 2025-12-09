#!/usr/bin/env python

import os
import json
import time
import logging
import warnings
import numpy as np
import xarray as xr
import torch

from utils import Config
from data import DataPrep
from models import build_model_from_config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

CFG = Config()
FILEDIR = CFG.filedir
MODELDIR = CFG.modeldir
RESULTSDIR = CFG.resultsdir
FIELDVARS = CFG.fieldvars
LOCALVARS = CFG.localvars
TARGETVAR = CFG.targetvar
LATRANGE = CFG.latrange
LONRANGE = CFG.lonrange
MODEL_CFGS = CFG.models
SEED = CFG.seed
WORKERS = CFG.workers
BATCHSIZE = CFG.batchsize

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP_ENABLED = (DEVICE == 'cuda')
if DEVICE == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

torch.manual_seed(SEED)
np.random.seed(SEED)

statspath = os.path.join(FILEDIR, 'stats.json')
with open(statspath, 'r', encoding='utf-8') as f:
    STATS = json.load(f)


def load_checkpoint(model, runname, modeldir=MODELDIR):
    '''
    Purpose: Load best checkpoint for a given run name.
    Args:
    - model (torch.nn.Module): model instance
    - runname (str): model run name
    - modeldir (str): directory containing checkpoints (defaults to MODELDIR)
    Returns:
    - bool: True if checkpoint loaded successfully, False otherwise
    '''
    filename = f'{runname}_best.pth'
    filepath = os.path.join(modeldir, filename)
    if not os.path.exists(filepath):
        logger.error(f'Checkpoint not found: {filepath}')
        return False
    state = torch.load(filepath, map_location='cpu')
    model.load_state_dict(state)
    logger.info(f'   Loaded checkpoint from {filename}')
    return True


def denormalize_precip(ynorm, targetvar=TARGETVAR, stats=STATS):
    '''
    Purpose: Undo z-score + log1p normalization using pre-loaded stats.
    Args:
    - ynorm (np.ndarray): normalized predictions (nsamples,)
    - targetvar (str): target variable name used to construct keys in stats (defaults to TARGETVAR)
    - stats (dict): loaded statistics from stats.json (defaults to STATS)
    Returns:
    - np.ndarray: denormalized predictions (nsamples,)
    '''
    mean = float(stats[f'{targetvar}_mean'])
    std = float(stats[f'{targetvar}_std'])
    ylog = ynorm * std + mean
    y = np.expm1(ylog)
    return y


def predict_precip(model, loader, centers, targettemplate, quadweights, device=DEVICE):
    '''
    Purpose: Run model, denormalize, and reconstruct full (lat,lon,time) field.
    Args:
    - model (torch.nn.Module): trained model
    - loader (DataLoader): data loader for evaluation
    - centers (list[tuple[int,int,int]]): (latidx, lonidx, timeidx) for each sample
    - targettemplate (torch.Tensor): (nlats, nlons, ntimes) template for reconstruction
    - quadweights (torch.Tensor): quadrature weights for kernel models
    - device (str): device to run inference on (defaults to DEVICE)
    Returns:
    - np.ndarray: reconstructed precipitation field (nlats, nlons, ntimes)
    '''
    model = model.to(device)
    quadweights = quadweights.to(device) if quadweights is not None else None
    model.eval()
    preds = []
    start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            for batch in loader:
                patch = batch['patch'].to(device, non_blocking=True)
                local = batch.get('local', None)
                if local is not None:
                    local = local.to(device, non_blocking=True)
                if hasattr(model, 'kernellayer'):
                    ypred = model(patch, local, quadweights)
                else:
                    ypred = model(patch, local)
                preds.append(ypred.squeeze(-1).cpu().numpy())
    duration = time.time() - start
    logger.info(f'   Inference time: {duration:.1f} s')
    ynorm = np.concatenate(preds, axis=0)
    yphys = denormalize_precip(ynorm)
    arr = np.full(targettemplate.shape, np.nan, dtype=np.float64)
    for idx, (ilat, ilon, itime) in enumerate(centers):
        arr[ilat, ilon, itime] = yphys[idx]
    return arr


def save_precip(arr, templateda, runname, splitname, resultsdir=RESULTSDIR):
    '''
    Purpose: Save predicted precipitation to NetCDF on the target grid.
    Args:
    - arr (np.ndarray): precipitation predictions (nlats, nlons, ntimes)
    - templateda (xr.DataArray): template DataArray with correct coordinates/dims
    - runname (str): model run name
    - splitname (str): 'valid' | 'test'
    - resultsdir (str): output directory (defaults to RESULTSDIR)
    Returns:
    - bool: True if save successful, False otherwise
    '''
    os.makedirs(resultsdir, exist_ok=True)
    da = xr.DataArray(
        arr,
        dims=templateda.dims,
        coords=templateda.coords,
        name='pr')
    da.attrs = dict(long_name='NN-predicted precipitation rate', units='mm/hr')
    filename = f'nn_{runname}_{splitname}_pr.nc'
    filepath = os.path.join(resultsdir, filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        da.to_netcdf(filepath, engine='h5netcdf')
        with xr.open_dataset(filepath, engine='h5netcdf') as _:
            pass
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False


def parse_args():
    '''
    Purpose: Parse command-line arguments for evaluation script.
    Returns:
    - argparse.Namespace: parsed arguments
    '''
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
        choices=['valid', 'test'],
        help='Which split to evaluate.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    requested = [m.strip() for m in args.models.split(',')] if args.models != 'all' else None
    splitname = args.split
    logger.info(f'Evaluating models on {splitname} split...')
    splitdata = DataPrep.prepare_splits(FILEDIR, FIELDVARS, LOCALVARS, TARGETVAR, [splitname])
    targetda = splitdata[splitname]['ds'][TARGETVAR]
    nfieldvars = len(FIELDVARS)
    nlocalvars = len(LOCALVARS)
    nlevs = splitdata[splitname]['fielddata'].shape[3]
    cachedconfig = None
    cachedresult = None
    for modelcfg in MODEL_CFGS:
        modelname = modelcfg['name']
        if (requested is not None) and (modelname not in requested):
            continue
        logger.info(f'Evaluating model {modelname} ({modelcfg["type"]}) on {splitname}')
        patchcfg = modelcfg['patch']
        uselocal = modelcfg.get('uselocal', True)
        currentconfig = (patchcfg['radius'], patchcfg['maxlevs'], patchcfg['timelag'], uselocal)
        if currentconfig == cachedconfig:
            logger.info('   Reusing cached dataset and loader')
            result = cachedresult
        else:
            result = DataPrep.build_datasets_and_loaders(
                splitdata, patchcfg, uselocal, LATRANGE, LONRANGE, BATCHSIZE, WORKERS, DEVICE, nlevs)
            logger.info(f'   {splitname} samples: {len(result["centers"][splitname])}')
            cachedconfig = currentconfig
            cachedresult = result
        model = build_model_from_config(modelcfg, result['patchshape'], nfieldvars, nlocalvars)
        if not load_checkpoint(model, modelname):
            logger.error(f'Skipping model {modelname} (no checkpoint).')
            continue
        yarr = predict_precip(
            model, result['loaders'][splitname], result['centers'][splitname],
            splitdata[splitname]['targetdata'], result['quadweights'], device=DEVICE)
        save_precip(yarr, targetda, modelname, splitname)
    logger.info('Finished evaluating selected models.')