#!/usr/bin/env python

import os 
import time
import torch
import wandb
import logging
import argparse
import numpy as np
from scripts.utils import Config
from scripts.data.inputs import InputDataModule
from scripts.models.architectures import ModelFactory 

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

config = Config()
SPLITDIR  = config.splitsdir 
MODELDIR  = config.modelsdir
FIELDVARS = config.fieldvars
LOCALVARS = config.localvars
TARGETVAR = config.targetvar
LATRANGE  = config.latrange
LONRANGE  = config.lonrange
PROJECT   = config.projectname 
SEED      = config.seed
BATCHSIZE = config.batchsize
WORKERS   = config.workers
EPOCHS    = config.epochs
LR        = config.learningrate
PATIENCE  = config.patience
CRITERION = config.criterion

def setup(seed=SEED):
    '''
    Purpose: Set random seeds for reproducibility and configure the compute device.
    Args:
    - seed (int): random seed for NumPy and PyTorch (defaults to SEED)
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
    parser = argparse.ArgumentParser(description='Train and validate NN models.')
    parser.add_argument('--models',type=str,default='all',help='Comma-separated list of model names to evaluate, or `all`.')
    args = parser.parse_args()
    return None if args.models=='all' else {name.strip() for name in args.models.split(',')}

def initialize(name,modelconfig,result,device,fieldvars=FIELDVARS,localvars=LOCALVARS):
    '''
    Purpose: Initialize a model instance from ModelFactory.build().
    Args:
    - name (str): model name
    - modelconfig (dict): model configuration
    - result (dict[str,object]): dictionary from InputDataModule.dataloaders()
    - device (str): device to use
    - fieldvars (list[str]): predictor field variable names (defaults to FIELDVARS)
    - localvars (list[str]): local input variable names (defaults to LOCALVARS)
    Returns:
    - torch.nn.Module: initialized model instance on 'device'
    '''
    patchshape = result['geometry'].shape()
    nfieldvars = len(fieldvars)
    nlocalvars = len(localvars)
    model = ModelFactory.build(name,modelconfig,patchshape,nfieldvars,nlocalvars)
    return model.to(device)
    
def save(name,state,kind,modeldir=MODELDIR):
    '''
    Purpose: Save best (lowest validation loss) model checkpoint, then verify by reopening.
    Args:
    - name (str): model name
    - state (dict): model.state_dict() to save
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

def fit(name,model,kind,result,uselocal,device,project=PROJECT,lr=LR,patience=PATIENCE,criterion=CRITERION,epochs=EPOCHS,modeldir=MODELDIR):
    '''
    Purpose: Train a model with early stopping and learning rate scheduling, then save the best checkpoint.
    Args:
    - name (str): model name
    - model (torch.nn.Module): initialized model instance
    - kind (str): 'baseline' | 'nonparametric' | 'parametric'
    - result (dict[str,object]): dictionary from InputDataModule.dataloaders()
    - uselocal (bool): whether to use local inputs
    - device (str): device to use
    - project (str): project name for Weights & Biases logging
    - lr (float): initial learning rate (defaults to LR)
    - patience (int): early stopping patience (defaults to PATIENCE)
    - criterion (str): loss function name (defaults to CRITERION)
    - epochs (int): maximum number of epochs (defaults to EPOCHS)
    - modeldir (str): output directory for checkpoints (defaults to MODELDIR)
    '''
    trainloader = result['loaders']['train']
    validloader = result['loaders']['valid']
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,max_lr=lr,epochs=epochs,steps_per_epoch=len(trainloader),pct_start=0.1,anneal_strategy='cos')
    wandb.init(project=project,name=name,
               config={
                   'Epochs':epochs,
                   'Batch size':trainloader.batch_size,
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
            
            fieldpatch   = batch['fieldpatch'].to(device)
            localvalues  = batch['localvalues'].to(device) if uselocal else None
            targetvalues = batch['targetvalues'].to(device)

            if hasattr(model,'intkernel'):
                quadpatch  = batch['quadpatch'].to(device)
                dareapatch = batch['dareapatch'].to(device)
                dlevpatch  = batch['dlevpatch'].to(device)
                dtimepatch = batch['dtimepatch'].to(device)

            optimizer.zero_grad()

            if hasattr(model,'intkernel'):
                outputvalues = model(fieldpatch,quadpatch,dareapatch,dlevpatch,dtimepatch,localvalues)
            else:
                outputvalues = model(fieldpatch,localvalues)

            loss = criterion(outputvalues,targetvalues)
            loss.backward()
            optimizer.step()
            scheduler.step() 
            totalloss += loss.item()*len(targetvalues)

        trainloss = totalloss/len(trainloader.dataset)

        model.eval()
        totalloss = 0.0
        with torch.no_grad():
            for batch in validloader:

                fieldpatch   = batch['fieldpatch'].to(device)
                localvalues  = batch['localvalues'].to(device) if uselocal else None
                targetvalues = batch['targetvalues'].to(device)

                if hasattr(model,'intkernel'):
                    quadpatch  = batch['quadpatch'].to(device)
                    dareapatch = batch['dareapatch'].to(device)
                    dlevpatch  = batch['dlevpatch'].to(device)
                    dtimepatch = batch['dtimepatch'].to(device)

                if hasattr(model,'intkernel'):
                    outputvalues = model(fieldpatch,quadpatch,dareapatch,dlevpatch,dtimepatch,localvalues)
                else:
                    outputvalues = model(fieldpatch,localvalues)

                loss = criterion(outputvalues,targetvalues)
                totalloss += loss.item()*len(targetvalues)

        validloss = totalloss/len(validloader.dataset)

        if validloss<bestloss:
            beststate = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            bestloss  = validloss
            bestepoch = epoch
            noimprove = 0
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
        save(name,beststate,kind,modeldir)
    wandb.finish()
    
if __name__=='__main__':
    logger.info('Spinning up...')
    device = setup()
    models = parse()
    logger.info('Preparing data splits...')
    splitdata    = InputDataModule.prepare(['train','valid'],FIELDVARS,LOCALVARS,TARGETVAR,SPLITDIR)
    cachedconfig = None
    cachedresult = None
    for modelconfig in config.models:
        name = modelconfig['name']
        kind = modelconfig['kind']
        if models is not None and name not in models:
            continue
        logger.info(f'Training `{name}`...')
        model = None
        patchconfig   = modelconfig['patch']
        uselocal      = modelconfig['uselocal']
        currentconfig = (patchconfig['radius'],patchconfig['maxlevs'],patchconfig['timelag'],uselocal)
        if currentconfig==cachedconfig:
            result = cachedresult
        else:
            result = InputDataModule.dataloaders(splitdata,patchconfig,uselocal,LATRANGE,LONRANGE,BATCHSIZE,WORKERS,device)
            cachedconfig = currentconfig
            cachedresult = result
        model = initialize(name,modelconfig,result,device)
        fit(name,model,kind,result,uselocal,device)
        del model