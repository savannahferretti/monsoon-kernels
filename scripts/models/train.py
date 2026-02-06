#!/usr/bin/env python

import os
import torch
import logging
import argparse
import numpy as np
from scripts.utils import Config
from scripts.data.classes import PatchDataLoader
from scripts.models.classes import ModelFactory,Trainer

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
    Purpose: Parse command-line arguments for running the training script.
    Returns:
    - set[str] | None: model names to run, or None if all models should be run
    '''
    parser = argparse.ArgumentParser(description='Train and validate NN models')
    parser.add_argument('--models',type=str,default='all',help='Comma-separated list of model names to evaluate, or `all`')
    args = parser.parse_args()
    return None if args.models=='all' else {name.strip() for name in args.models.split(',')}

def initialize(name,modelconfig,result,device,fieldvars,localvars):
    '''
    Purpose: Initialize a model instance from ModelFactory.
    Args:
    - name (str): model name
    - modelconfig (dict): model configuration
    - result (dict[str,object]): dictionary from InputDataModule.dataloaders()
    - device (str): device to use
    - fieldvars (list[str]): predictor field variable names
    - localvars (list[str]): local input variable names
    Returns:
    - torch.nn.Module: initialized model instance on device
    '''
    patchshape = result['geometry'].shape()
    nfieldvars = len(fieldvars)+1  
    nlocalvars = len(localvars)
    model = ModelFactory.build(name,modelconfig,patchshape,nfieldvars,nlocalvars)
    return model.to(device)

if __name__=='__main__':
    config = Config()
    logger.info('Spinning up...')
    models = parse()
    logger.info('Preparing data splits...')
    splitdata  = PatchDataLoader.prepare(['train','valid'],config.fieldvars,config.localvars,config.targetvar,config.splitsdir)
    maxradius  = max(m['patch']['radius'] for m in config.models)
    maxtimelag = max(m['patch']['timelag'] for m in config.models)
    cachedconfig = None
    cachedresult = None
    for modelconfig in config.models:
        name = modelconfig['name']
        kind = modelconfig['kind']
        if models is not None and name not in models:
            continue
        seeds = modelconfig.get('seeds',config.seeds)
        for seed in seeds:
            modelname = f'{name}_{seed}'
            if os.path.exists(os.path.join(config.modelsdir,f'{modelname}.pth')):
                logger.info(f'Skipping `{modelname}`, checkpoint already exists')
                continue
            logger.info(f'Training `{modelname}`...')
            device        = setup(seed)
            uselocal      = modelconfig['uselocal']
            patchconfig   = modelconfig['patch']
            currentconfig = (patchconfig['radius'],patchconfig['levmode'],patchconfig['timelag'],uselocal)
            if currentconfig==cachedconfig:
                result = cachedresult
            else:
                result = PatchDataLoader.dataloaders(splitdata,patchconfig,uselocal,config.latrange,config.lonrange,config.batchsize,config.workers,device,maxradius,maxtimelag)
                cachedconfig = currentconfig
                cachedresult = result
            model = initialize(name,modelconfig,result,device,config.fieldvars,config.localvars)
            trainer = Trainer(
                model=model,
                trainloader=result['loaders']['train'],
                validloader=result['loaders']['valid'],
                device=device,
                modeldir=config.modelsdir,
                project=config.projectname,
                seed=seed,
                lr=config.learningrate,
                patience=config.patience,
                criterion=config.criterion,
                epochs=config.epochs,
                use_amp=True,
                grad_accum_steps=1,
                compile_model=False)
            trainer.fit(name,kind,uselocal)
            del model,trainer
