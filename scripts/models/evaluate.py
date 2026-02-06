#!/usr/bin/env python

import os
import torch
import logging
import argparse
import numpy as np
import xarray as xr
from scripts.utils import Config
from scripts.data.classes import PatchDataLoader,PredictionWriter
from scripts.models.classes import ModelFactory,Inferencer

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def setup(seed):
    '''
    Purpose: Set random seeds for reproducibility and configure compute device.
    Args:
    - seed (int): random seed for NumPy and PyTorch
    Returns:
    - str: device to use
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    return device

def parse():
    '''
    Purpose: Parse command-line arguments for running the evaluation script.
    Returns:
    - tuple[set[str] | None, str]: model names to run or None if all, and split name
    '''
    parser = argparse.ArgumentParser(description='Evaluate NN models.')
    parser.add_argument('--models',type=str,default='all',help='Comma-separated list of model names to evaluate, or `all`.')
    parser.add_argument('--split',type=str,default='test',help='Split to evaluate: train|valid|test (default: test)')
    args = parser.parse_args()
    models = None if args.models=='all' else {name.strip() for name in args.models.split(',') if name.strip()}
    return models,args.split

def load(name,modelconfig,result,device,fieldvars,localvars,modeldir,seed):
    '''
    Purpose: Initialize a model from ModelFactory and load weights from a saved checkpoint.
    Args:
    - name (str): model name
    - modelconfig (dict): model configuration
    - result (dict[str,object]): dictionary from PatchDataLoader.dataloaders()
    - device (str): device to use
    - fieldvars (list[str]): predictor field variable names
    - localvars (list[str]): local input variable names
    - modeldir (str): directory containing checkpoints
    - seed (int): random seed used during training
    Returns:
    - torch.nn.Module: model with loaded state_dict on device, or None if checkpoint not found
    '''
    patchshape = result['geometry'].shape()
    nfieldvars = len(fieldvars)+1
    nlocalvars = len(localvars)
    filepath = os.path.join(modeldir,f'{name}_{seed}.pth')
    if not os.path.exists(filepath):
        logger.error(f'   Checkpoint not found: {filepath}')
        return None
    model = ModelFactory.build(name,modelconfig,patchshape,nfieldvars,nlocalvars)
    state = torch.load(filepath,map_location='cpu')
    model.load_state_dict(state)
    return model.to(device)

if __name__=='__main__':
    config = Config()
    logger.info('Spinning up...')
    device = setup(config.seeds[0])
    models,split = parse()
    logger.info('Preparing evaluation split...')
    splitdata  = PatchDataLoader.prepare([split],config.fieldvars,config.localvars,config.targetvar,config.splitsdir)
    maxradius  = max(m['patch']['radius'] for m in config.models)
    maxtimelag = max(m['patch']['timelag'] for m in config.models)
    writer = PredictionWriter(config.fieldvars)
    cachedconfig = None
    cachedresult = None
    for modelconfig in config.models:
        name = modelconfig['name']
        if models is not None and name not in models:
            continue
        seeds = modelconfig.get('seeds',config.seeds)
        if os.path.exists(os.path.join(config.predsdir,f'{name}_{split}_predictions.nc')):
            logger.info(f'Skipping `{name}`, predictions already exist')
            continue
        logger.info(f'Evaluating `{name}` across {len(seeds)} seed(s)...')
        uselocal      = modelconfig['uselocal']
        patchconfig   = modelconfig['patch']
        currentconfig = (patchconfig['radius'],patchconfig['levmode'],patchconfig['timelag'],uselocal)
        if currentconfig==cachedconfig:
            result = cachedresult
        else:
            result = PatchDataLoader.dataloaders(splitdata,patchconfig,uselocal,config.latrange,config.lonrange,config.batchsize,config.workers,device,maxradius,maxtimelag)
            cachedconfig = currentconfig
            cachedresult = result
        haskernel  = modelconfig['kind']!='baseline'
        nonparam   = modelconfig['kind']=='nonparametric'
        kerneldims = tuple(modelconfig.get('kerneldims',modelconfig.get('kerneldict',{}).keys()))
        centers = result['centers'][split]
        refda   = splitdata[split]['refda']
        allpreds      = []
        allcomponents = []
        for seedidx,seed in enumerate(seeds):
            logger.info(f'   Seed {seedidx+1}/{len(seeds)} ({seed})...')
            model = load(name,modelconfig,result,device,config.fieldvars,config.localvars,config.modelsdir,seed)
            if model is None:
                logger.error(f'   Failed to load model for seed {seed}, skipping this model entirely')
                break
            inferencer = Inferencer(model,result['loaders'][split],device)
            preds      = inferencer.predict(uselocal,haskernel)
            arr,meta   = writer.to_array(preds,'predictions',centers=centers,refda=refda)
            allpreds.append(arr)
            if haskernel:
                components = inferencer.extract_weights(nonparam)
                seedcomps  = []
                for comp in components:
                    warr,wmeta = writer.to_array(comp,'weights',kerneldims=kerneldims,nonparam=nonparam)
                    seedcomps.append(warr)
                allcomponents.append(seedcomps)
            del model,inferencer
        else:
            logger.info('   Combining results from all seeds...')
            predstack = np.stack(allpreds,axis=-1)
            logger.info('   Formatting and saving predictions...')
            ds = writer.to_dataset(predstack,meta,refda=refda,seedaxis=True)
            writer.save(name,ds,'predictions',split,config.predsdir)
            del predstack,ds
            if haskernel:
                ncomps  = len(allcomponents[0])
                stacked = [np.stack([s[i] for s in allcomponents],axis=-1) for i in range(ncomps)]
                logger.info('   Formatting and saving normalized kernel weights...')
                refds = xr.open_dataset(os.path.join(config.splitsdir,'valid.h5'),engine='h5netcdf')
                ds    = writer.to_dataset(stacked,wmeta,refds=refds,seedaxis=True)
                writer.save(name,ds,'weights',split,config.weightsdir)
                del stacked,ds
