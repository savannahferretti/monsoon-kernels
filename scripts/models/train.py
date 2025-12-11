#!/usr/bin/env python

import os
import time
import torch
import wandb
import logging
import argparse
import numpy as np
from utils import Config
from dataset import DataModule
from models import BaselineNN,KernelNN,ModelFactory
from kernels import NonparametricKernelLayer,ParametricKernelLayer

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

config = Config()
SPLITDIR  = config.splitdir
MODELDIR  = config.modeldir
FIELDVARS = config.fieldvars
LOCALVARS = config.localvars
TARGETVAR = config.targetvar
LATRANGE  = config.latrange
LONRANGE  = config.lonrange
PROJECT   = config.project
SEED      = config.seed
WORKERS   = config.workers
EPOCHS    = config.epochs
BATCHSIZE = config.batchsize
LR        = config.learningrate
PATIENCE  = config.patience
CRITERION = config.criterion

def initialize(name,modelconfig,result,device,fieldvars=FIELDVARS,localvars=LOCALVARS,modeldir=MODELDIR):
    '''
    Purpose: Initialize a model instance from ModelFactory.build().
    Args:
    - name (str): model name
    - modelconfig (dict): model configuration
    - result (dict[str,object]): dictionary from DataModule.dataloaders()
    - device (str): device to use ('cpu' | 'cuda')
    - fieldvars (int): predictor field variable names (defaults to FIELDVARS)
    - localvars (int): local input variable names (defaults to LOCALVARS)
    - modeldir (str): directory containing checkpoints (defaults to MODELDIR)
    Returns:
    - torch.nn.Module: initialized model instance on 'device'
    '''
    patchshape = result['geometry'].shape
    nfieldvars = len(fieldvars)
    nlocalvars = len(localvars)
    model = ModelFactory.build(name,modelconfig,patchshape,nfieldvars,nlocalvars)
    return model.to(device)
    
def save(state,name,kind,modeldir=MODELDIR):
    '''
    Purpose: Save best (lowest validation loss) model checkpoint, then verify by reopening.
    Args:
    - state (dict): model.state_dict() to save
    - name (str): model name
    - kind (str): 'baseline' | 'nonparametric' | 'parametric'
    - modeldir (str): output directory (defaults to MODELDIR)
    Returns:
    - bool: True if save successful, False otherwise
    '''
    savedir = os.path.join(modeldir,kind)
    os.makedirs(savedir,exist_ok=True)
    filename = f'{name}.pth'
    filepath = os.path.join(savedir,filename)
    logger.info(f'      Attempting to save {filename}...')
    try:
        torch.save(state,filepath)
        _ = torch.load(filepath,map_location='cpu')
        logger.info('         File write successful')
        return True
    except Exception:
        logger.exception('         Failed to save or verify')
        return False

def fit(name,model,kind,result,uselocal,device,
        project=PROJECT,batchsize=BATCHSIZE,lr=LR,patience=PATIENCE,criterion=CRITERION,epochs=EPOCHS,modeldir=MODELDIR):
    '''
    Purpose: Train a model with early stopping and learning rate scheduling, then save the best checkpoint.
    Args:
    - name (str): model name
    - model (torch.nn.Module): initialized model instance
    - kind (str): 'baseline' | 'nonparametric' | 'parametric'
    - result (dict[str,object]): dictionary from DataModule.dataloaders()
    - uselocal (bool): whether to use local inputs
    - device (str): device to use
    - project (str): project name for Weights & Biases logging
    - batchsize (int): batch size (defaults to BATCHSIZE)
    - lr (float): learning rate (defaults to LR)
    - patience (int): early stopping patience (defaults to PATIENCE)
    - criterion (str): loss function name (defaults to CRITERION)
    - epochs (int): maximum number of epochs (defaults to EPOCHS)
    - modeldir (str): directory to save model checkpoints (defaults to MODELDIR)
    '''
    trainloader = result['loaders']['train']
    validloader = result['loaders']['valid']
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,max_lr=lr,epochs=epochs,steps_per_epoch=len(trainloader),pct_start=0.2,anneal_strategy='cos',div_factor=10,final_div_factor=100)
    wandb.init(project=project,name=name,
               config={
                   'Epochs':epochs,
                   'Batch size':batchsize,
                   'Initial learning rate':lr,
                   'Early stopping patience':patience,
                   'Loss function':criterion,
                   'Number of parameters':model.nparams,
                   'Device':device})
    criterion = getattr(torch.nn,criterion)()
    beststate = None
    bestloss  = float('inf')
    bestepoch = 0
    noimprove = 0
    starttime = time.time()
    for epoch in range(1,epochs+1):
        model.train()
        totalloss = 0.0
        for batch in trainloader:
            patch  = batch['patch'].to(device)
            target = batch['target'].to(device)
            local  = batch['local'].to(device) if uselocal else None
            optimizer.zero_grad()
            if hasattr(model,'kernellayer'):
                quad   = result['quad'].to(device)
                output = model(patch,quad,local)
            else:
                output = model(patch,local)
            loss   = criterion(output,target)
            loss.backward()
            optimizer.step()
            totalloss += loss.item()*len(target)
        trainloss = totalloss/len(trainloader.dataset)
        model.eval()
        totalloss = 0.0
        with torch.no_grad():
            for batch in validloader:
                patch  = batch['patch'].to(device)
                target = batch['target'].to(device)
                local  = batch['local'].to(device) if uselocal else None
                if hasattr(model,'kernellayer'):
                    quad   = result['quad'].to(device)
                    output = model(patch,quad,local)
                else:
                    output = model(patch,local)
                loss   = criterion(output,target)
                totalloss += loss.item()*len(target)
        validloss = totalloss/len(validloader.dataset)
        if validloss<bestloss:
            bestloss  = validloss
            bestepoch = epoch
            noimprove = 0
            beststate = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        else:
            noimprove += 1
        wandb.log({
            'Epoch':epoch,
            'Training loss':trainloss,
            'Validation loss':validloss,
            'Learning rate':optimizer.param_groups[0]['lr']})
        if noimprove>=patience:
            break
    duration = time.time()-starttime
    wandb.run.summary.update({
        'Best model at epoch':bestepoch,
        'Best validation loss':bestloss,
        'Training duration (s)':duration,
        'Stopped early':noimprove>=patience})
    if beststate is not None:
        save(beststate,name,kind)
    wandb.finish()

def parse():
    '''
    Purpose: Parse command-line arguments for running the training script.
    Returns:
    - argparse.Namespace: parsed arguments
    '''
    parser = argparse.ArgumentParser(description='Train and validate NN precipitation models.')
    parser.add_argument('--models',type=str,default='all',help='Comma-separated list of model names to evaluate, or `all`.')
    return parser.parse_args()
    
if __name__=='__main__':
    logger.info('Setting random seed...')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    logger.info('Determining device type...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    logger.info('Parsing arguments...')
    args   = parse()
    models = [m.strip() for m in args.models.split(',')] if args.models!='all' else None
    logger.info('Preparing data splits...')
    splitdata    = DataModule.prepare(['train','valid'],FIELDVARS,LOCALVARS,TARGETVAR,SPLITDIR)
    cachedconfig = None
    cachedresult = None
    for name,modelconfig in config.models.items():
        name = modelconfig['name']
        kind = modelconfig['kind']
        if name not in models:
            continue
        logger.info(f'Running `{name}`...')
        patchconfig   = modelconfig['patch']
        uselocal      = modelconfig['uselocal']
        currentconfig = (patchconfig['radius'],patchconfig['maxlevs'],patchconfig['timelag'],uselocal)
        if currentconfig==cachedconfig:
            logger.info('   Reusing cached datasets and loaders...')
            result = cachedresult
        else:
            logger.info('   Building new datasets and loaders....')
            result = DataModule.dataloaders(splitdata,patchconfig,uselocal,LATRANGE,LONRANGE,BATCHSIZE,WORKERS,device)
            cachedconfig = currentconfig
            cachedresult = result
        logger.info('   Initializing model....')
        model = initialize(name,modelconfig,result,device)
        logger.info('   Starting training....')
        fit(name,model,kind,result,uselocal,device)
        del model