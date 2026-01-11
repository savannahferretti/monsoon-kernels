#!/usr/bin/env python

import os
import time
import torch
import logging
import argparse
import numpy as np
import xarray as xr
from scripts.utils import Config
from scripts.data.classes import PatchDataLoader,PredictionWriter
from scripts.models.classes import ModelFactory

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

config = Config()
SPLITDIR   = config.splitsdir
PREDSDIR   = config.predsdir
FEATSDIR   = config.featsdir
WEIGHTSDIR = config.weightsdir
MODELDIR   = config.modelsdir
FIELDVARS  = config.fieldvars
LOCALVARS  = config.localvars
TARGETVAR  = config.targetvar
LATRANGE   = config.latrange
LONRANGE   = config.lonrange
SEED       = config.seed
BATCHSIZE  = config.batchsize
WORKERS    = config.workers

out = PredictionWriter(FIELDVARS)

def setup(seed=SEED):
    '''
    Purpose: Set random seeds for reproducibility and configure the compute device.
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
    - tuple(set[str] | None,str): model names to run, or None if all models should be run, and split to use for inference
    '''
    parser = argparse.ArgumentParser(description='Evaluate NN models.')
    parser.add_argument('--models',type=str,default='all',help='Comma-separated list of model names to evaluate, or `all`.')
    parser.add_argument('--split',type=str,default='test',help='Split to evaluate: train|valid|test (default: test)')
    args = parser.parse_args()
    models = None if args.models=='all' else {name.strip() for name in args.models.split(',') if name.strip()}
    return models,args.split
    
def load(name,modelconfig,result,device,fieldvars=FIELDVARS,localvars=LOCALVARS,modeldir=MODELDIR):
    '''
    Purpose: Initialize a model instance from ModelFactory.build() and populate with weights from a saved checkpoint.
    Args:
    - name (str): model name
    - modelconfig (dict): model configuration
    - result (dict[str,object]): dictionary from PatchDataLoader.dataloaders()  
    - device (str): device to use
    - fieldvars (list[str]): predictor field variable names (defaults to FIELDVARS)
    - localvars (list[str]): local input variable names (defaults to LOCALVARS)
    - modeldir (str): directory containing checkpoints (defaults to MODELDIR)
    Returns:
    - torch.nn.Module: model with loaded state_dict on 'device' or None if checkpoint not found
    '''
    kind     = modelconfig['kind']
    filedir  = os.path.join(modeldir,kind)
    filename = f'{name}.pth'
    filepath = os.path.join(filedir,filename)
    if not os.path.exists(filepath):
        logger.error(f'   Checkpoint not found: {filepath}')
        return None
    patchshape = result['geometry'].shape()
    nfieldvars = 2*len(fieldvars)
    nlocalvars = len(localvars)
    model = ModelFactory.build(name,modelconfig,patchshape,nfieldvars,nlocalvars)
    if hasattr(model,"intkernel") and hasattr(model.intkernel,"kernel") and (model.intkernel.kernel is None):
        batch = next(iter(result["loaders"]["valid"]))
        with torch.no_grad():
            fieldpatch = batch["fieldpatch"].to(device,non_blocking=True)
            dareapatch = batch["dareapatch"].to(device,non_blocking=True)
            dlevpatch  = batch["dlevpatch"].to(device,non_blocking=True)
            dtimepatch = batch["dtimepatch"].to(device,non_blocking=True)
            dlevfull   = batch["dlevfull"].to(device,non_blocking=True)
            _ = model.intkernel(fieldpatch,dareapatch,dlevpatch,dtimepatch,dlevfull)
    state = torch.load(filepath,map_location='cpu')
    model.load_state_dict(state)
    return model.to(device)

def inference(model,split,result,uselocal,device):
    '''
    Purpose: Run inference for predictions, and if present, also collect kernel-integrated features and normalized weights.
    Notes:
    - Predictions are the forward pass outputs.
    - Features are pulled from model.intkernel.features after each forward pass.
    - Normalized weights are pulled from model.intkernel.weights (captured once after first forward pass).
    Args:
    - model (torch.nn.Module): trained model instance
    - split (str): 'train' | 'valid' | 'test'
    - result (dict[str,object]): dictionary from PatchDataLoader.dataloaders()
    - uselocal (bool): whether to use local inputs
    - device (str): device to use
    Returns:
    - dict[str,object]: dictionary with NumPy arrays and kernel metadata
    '''
    dataloader = result['loaders'][split]
    havekernel = hasattr(model,'intkernel')
    nonparam = bool(havekernel and ('nonparametric' in model.intkernel.__class__.__name__.lower()))
    nkernels = int(getattr(model,'nkernels',1)) if havekernel else 1
    kerneldims = tuple(getattr(model,'kerneldims',())) if havekernel else tuple()
    model.eval()
    predslist = []
    featslist = []
    weights = None
    with torch.no_grad():
        for batch in dataloader:
            fieldpatch = batch['fieldpatch'].to(device)
            localvalues = batch['localvalues'].to(device) if (uselocal and 'localvalues' in batch) else None
            if havekernel:
                dareapatch = batch['dareapatch'].to(device)
                dlevpatch = batch['dlevpatch'].to(device)
                dtimepatch = batch['dtimepatch'].to(device)
                dlevfull = batch['dlevfull'].to(device)
                outputvalues = model(fieldpatch,dareapatch,dlevpatch,dtimepatch,dlevfull,localvalues)
                if weights is None:
                    if model.intkernel.weights is None:
                        raise RuntimeError('`model.intkernel.weights` was not populated during forward pass')
                    weights = model.intkernel.weights.detach().cpu().numpy()
                if model.intkernel.features is not None:
                    featslist.append(model.intkernel.features.detach().cpu().numpy())
            else:
                outputvalues = model(fieldpatch,localvalues)
            predslist.append(outputvalues.detach().cpu().numpy())
    preds = np.concatenate(predslist,axis=0).astype(np.float32)
    feats = np.concatenate(featslist,axis=0).astype(np.float32) if featslist else None
    weights = weights.astype(np.float32) if weights is not None else None
    return {
        'predictions':preds,
        'features':feats,
        'weights':weights,
        'havekernel':havekernel,
        'nonparam':nonparam,
        'nkernels':nkernels,
        'kerneldims':kerneldims}

if __name__=='__main__':
    logger.info('Spinning up...')
    device       = setup()
    models,split = parse()
    logger.info('Preparing evaluation split...')
    splitdata = PatchDataLoader.prepare([split],FIELDVARS,LOCALVARS,TARGETVAR,SPLITDIR)
    maxradius = max(m['patch']['radius'] for m in config.models)
    maxtimelag = max(m['patch']['timelag'] for m in config.models)
    logger.info(f'Common domain constraints: maxradius={maxradius}, maxtimelag={maxtimelag}')
    cachedconfig = None
    cachedresult = None
    for modelconfig in config.models:
        name = modelconfig['name']
        kind = modelconfig['kind']
        if models is not None and name not in models:
            continue

        logger.info(f'Evaluating `{name}`...')

        patchconfig   = modelconfig['patch']
        uselocal      = modelconfig['uselocal']
        currentconfig = (patchconfig['radius'],patchconfig['levmode'],patchconfig['timelag'],uselocal)

        if currentconfig==cachedconfig:
            result = cachedresult
        else:
            result = PatchDataLoader.dataloaders(splitdata,patchconfig,uselocal,LATRANGE,LONRANGE,BATCHSIZE,WORKERS,device,maxradius,maxtimelag)
            cachedconfig = currentconfig
            cachedresult = result

        model = load(name,modelconfig,result,device)
        if model is None:
            continue

        info = inference(model,split,result,uselocal,device)
        centers = result['centers'][split]
        refda = splitdata[split]['refda']
        patchshape = result['geometry'].shape()
        logger.info('   Formatting/saving predictions...')
        arr,meta = out.to_array(info['predictions'],'predictions',
            centers=centers,refda=refda,nkernels=info['nkernels'],nonparam=info['nonparam'])
        ds = out.to_dataset(arr,meta,refda=refda,nkernels=info['nkernels'])
        out.save(name,ds,'predictions',split,PREDSDIR)
        del arr,meta,ds
        if info['havekernel']:
            logger.info('   Formatting/saving normalized kernel weights...')
            refds = xr.open_dataset(os.path.join(SPLITDIR,'valid.h5'),engine='h5netcdf')
            arr,meta = out.to_array(
                info['weights'],'weights',
                kerneldims=info['kerneldims'],
                nonparam=info['nonparam'])
            ds = out.to_dataset(arr,meta,refds=refds)
            out.save(name,ds,'weights',split,WEIGHTSDIR)
            del arr,meta,ds
            if info['features'] is not None:
                logger.info('   Formatting/saving kernel-integrated features...')
                arr,meta = out.to_array(
                    info['features'],'features',
                    centers=centers,refda=refda,nkernels=info['nkernels'],
                    kerneldims=info['kerneldims'],patchshape=patchshape,
                    nonparam=info['nonparam'])
                ds = out.to_dataset(arr,meta,refda=refda,nkernels=info['nkernels'])
                out.save(name,ds,'features',split,FEATSDIR)
                del arr,meta,ds
        del model