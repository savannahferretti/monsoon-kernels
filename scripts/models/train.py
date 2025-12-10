#!/usr/bin/env python

import os
import time
import torch
import wandb
import logging
import warnings 
import argparse
import numpy as np
from utils import Config
from data import DataPrep

logging.basicConfig(level=logging.INFO,ormat='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

config = Config()
SPLITDIR  = config.splitdir
MODELDIR  = config.modeldir
FIELDVARS = config.fieldvars
LOCALVARS = config.localvars
TARGETVAR = config.targetvar
LATRANGE  = config.latrange
LONRANGE  = config.lonrange
SEED      = config.seed
WORKERS   = config.workers
EPOCHS    = config.epochs
BATCHSIZE = config.batchsize
LR        = config.learningrate
PATIENCE  = config.patience
CRITERION = getattr(torch.nn,config.criterion)()

def save(state,name,modeldir=MODELDIR):
    '''
    Purpose: Save best (lowest validation loss) checkpoint for a run.
    Args:
    - state (dict): model state dictionary
    - name (str): model name
    - modeldir (str): output directory (defaults to MODELDIR)
    Returns:
    - bool: True if save successful, False otherwise
    '''
    os.makedirs(modeldir,exist_ok=True)
    filename = f'{name}.pth'
    filepath = os.path.join(modeldir,filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        torch.save(state,filepath)
        _ = torch.load(filepath,map_location='cpu')
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

def fit(model,name,trainloader,validloader,quadweights,device=DEVICE,lr=LR,patience=PATIENCE,batchsize=BATCHSIZE,criterion=CRITERION,epochs=EPOCHS,AMPENABLED):
    '''
    Purpose: Train a NN with early stopping and OneCycleLR, and save only the best checkpoint.
    Args:
    - model (torch.nn.Module): model to train
    - name (str): model name 
    - trainloader (DataLoader): training data loader
    - validloader (DataLoader): validation data loader
    - quadweights (torch.Tensor): quadrature weights for kernel models
    - device (str): device to train on (defaults to DEVICE)
    - lr (float): initial learning rate (defaults to LR)
    - batchsize (int): batch size (defaults to BATCHSIZE)
    - patience (int): early stopping patience (defaults to PATIENCE)
    - criterion (torch.nn.Module): loss function (defaults to CRITERION)
    - epochs (int): maximum number of epochs (defaults to EPOCHS)
    '''
    model = model.to(device)
    quadweights = quadweights.to(device) if quadweights is not None else None
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,epochs=epochs,steps_per_epoch=len(trainloader),pct_start=0.1,anneal_strategy='cos')
    scaler    = torch.cuda.amp.GradScaler(enabled=ampenabled)
    wandb.init(
        project='Chapter 2 Experiments',
        name=name,
        config={
            'Epochs': epochs,
            'Batch size':batchsizd,
            'Initial learning rate':lr,
            'Early stopping patience':patience,
            'Device':device,
            'AMP-enabled':ampenabled})
    bestloss = float('inf')
    beststate = None
    bestepoch = 0
    noimprove = 0
    starttime = time.time()
    for epoch in range(1,epochs+1):
        model.train()
        runningloss = 0.0
        for batch in trainloader:
            patch  = batch['patch'].to(device)
            local  = batch.get('local').to(device) if uselocal else None
            target = batch['target'].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=ampenabled):
                if hasattr(model,'kernellayer'):
                    output = model(patch,local,quadweights)
                else:
                    output = model(patch,local)
                loss = criterion(output,target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            runningloss += loss.item()*patch.size(0)
        trainloss = runningloss/len(trainloader.dataset)
        model.eval()
        runningloss = 0.0
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=ampenabled):
                for batch in validloader:
                    patch  = batch['patch'].to(device)
                    local  = batch.get('local').to(device) if uselocal else None
                    target = batch['target'].to(device)
                    if hasattr(model,'kernellayer'):
                        output = model(patch,local,quadweights)
                    else:
                        output = model(patch,local)
                    loss = criterion(output,target)
                    runningloss += loss.item()*patch.size(0)
        validloss = runningloss/len(validloader.dataset)
        wandb.log({
            'Training loss':trainloss,
            'Validation loss':validloss,
            'Learning rate':optimizer.param_groups[0]['lr']})
        if validloss<bestloss:
            bestloss  = validloss
            bestepoch = epoch
            noimprove = 0
            beststate = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        else:
            noimprove += 1
        if noimprove>=patience:
            logger.info(f'Early stopping at epoch {epoch:02d}')
            break
    duration = time.time()-starttime
    wandb.run.summary.update({
        'Best model at epoch':bestepoch,
        'Best validation loss':bestloss,
        'Total training epochs':epoch,
        'Training duration (s)':duration,
        'Stopped early':noimprove>=patience})
    if beststate is not None:
        save(beststate,name)
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

if __name__ == '__main__':
    logger.info('Setting random seed...')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    logger.info('Determining device type...')
    DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
    AMPENABLED = (DEVICE=='cuda')
    if AMPENABLED:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    logger.info('Parsing arguments...')
    args   = parse()
    models = [m.strip() for m in args.models.split(',')] if args.models!='all' else None
    logger.info('Loading training and validation splits...')
    splitdata = DataPrep.prepare(FILEDIR,FIELDVARS,LOCALVARS,TARGETVAR,['train','valid'])
    logger.info('Training selected models...')
    cachedconfig = None
    cachedresult = None
    for modelconfig in MODELCONFIGS:
        name = modelconfig['name']
        if (requested is not None) and (name not in requested):
            continue
        logger.info(f'Training `{name}`...')
        patchconfig = modelconfig['patchconfig']
        uselocal = modelconfig.get('uselocal',True)
        currentconfig = (patchconfig['radius'],patchconfig['maxlevs'],patchconfig['timelag'],uselocal)
        if currentconfig==cachedconfig:
            logger.info('   Reusing cached datasets and loaders...')
            result = cachedresult
        else:
            result = DataPrep.dataloader(splitdata,patchconfig,uselocal,LATRANGE,LONRANGE,BATCHSIZE,WORKERS,DEVICE,splitdata['train']['ds'].lev.size)
            cachedconfig = currentconfig
            cachedresult = result
        model = build(modelconfig,result['patchshape'],len(FIELDVARS),len(LOCALVARS))
        fit(model,name,result['loaders']['train'],result['loaders']['valid'],result['quadweights'])
    logger.info('Finished training selected models.')