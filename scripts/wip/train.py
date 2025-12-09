#!/usr/bin/env python

import os
import time
import torch
import wandb
import logging
import warnings
import argparse
import numpy as np
from io import IO
from utils import Config
from models import build
from torch.utils.data import DataLoader
from data import Data,Patch,SampleDataset

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

CFG = Config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = (DEVICE=='cuda')
if DEVICE=='cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)

def criterion(name):
    '''
    Purpose: Build torch loss function from string name in configs.json.
    Args:
    - name (str): loss function name (e.g., 'MSELoss')
    Returns:
    - torch.nn.Module: loss function instance
    '''
    if hasattr(torch.nn,name):
        return getattr(torch.nn,name)()
    raise ValueError(f'Unknown loss function: {name}')

def fit(model,name,trainloader,validloader):
    '''
    Purpose: Train model with early stopping, OneCycleLR scheduling, and W&B logging.
    Args:
    - model (torch.nn.Module): model to train
    - name (str): model name for checkpoints and logging
    - trainloader (DataLoader): training data loader
    - validloader (DataLoader): validation data loader
    '''
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=CFG.learningrate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=CFG.learningrate,
        epochs=CFG.epochs,steps_per_epoch=len(trainloader),pct_start=0.1,anneal_strategy='cos')
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    loss_fn = criterion(CFG.criterion)
    wandb.init(project='Monsoon Kernels',name=name,config={
        'Epochs':CFG.epochs,'Batch size':CFG.batchsize,'Learning rate':CFG.learningrate,
        'Patience':CFG.patience,'Loss function':CFG.criterion,'Device':DEVICE,'AMP':AMP})
    bestloss,beststate,bestepoch,noimprove = float('inf'),None,0,0
    start = time.time()
    for epoch in range(1,CFG.epochs+1):
        model.train()
        runningloss = 0.0
        for batch in trainloader:
            patch,target = batch['patch'].to(DEVICE,non_blocking=True),batch['target'].to(DEVICE,non_blocking=True)
            local = batch.get('local')
            if local is not None:
                local = local.to(DEVICE,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=AMP):
                ypred = model(patch,local).squeeze(-1)
                loss = loss_fn(ypred,target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            runningloss += loss.item()*patch.size(0)
        trainloss = runningloss/len(trainloader.dataset)
        model.eval()
        runningloss = 0.0
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=AMP):
            for batch in validloader:
                patch,target = batch['patch'].to(DEVICE,non_blocking=True),batch['target'].to(DEVICE,non_blocking=True)
                local = batch.get('local')
                if local is not None:
                    local = local.to(DEVICE,non_blocking=True)
                ypred = model(patch,local).squeeze(-1)
                loss = loss_fn(ypred,target)
                runningloss += loss.item()*patch.size(0)
        validloss = runningloss/len(validloader.dataset)
        wandb.log({'Epoch':epoch,'Train loss':trainloss,'Valid loss':validloss,'LR':optimizer.param_groups[0]['lr']})
        logger.info(f'Epoch {epoch:03d} | Train: {trainloss:.6e} | Valid: {validloss:.6e}')
        if validloss<bestloss:
            bestloss,bestepoch,noimprove = validloss,epoch,0
            beststate = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            noimprove += 1
        if noimprove>=CFG.patience:
            logger.info(f'Early stopping at epoch {epoch:03d}')
            break
    duration = time.time()-start
    logger.info(f'Best epoch: {bestepoch:03d} | Best valid loss: {bestloss:.6e} | Duration: {duration:.1f} s')
    wandb.run.summary.update({'Best epoch':bestepoch,'Best valid loss':bestloss,
        'Total epochs':epoch,'Duration (s)':duration,'Stopped early':noimprove>=CFG.patience})
    if beststate is not None:
        IO.save_model(beststate,name,CFG.modeldir)
    wandb.finish()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train NN precipitation models.')
    parser.add_argument('--models',type=str,default='all',
        help='Comma-separated model names from configs.json or "all"')
    args = parser.parse_args()
    requested = [m.strip() for m in args.models.split(',')] if args.models!='all' else None
    logger.info('Loading training and validation splits...')
    trainds,validds = IO.get_split('train',CFG.filedir),IO.get_split('valid',CFG.filedir)
    fieldtrain,localtrain,targettrain = Data.totensor(trainds,CFG.fieldvars,CFG.localvars,CFG.targetvar)
    fieldvalid,localvalid,targetvalid = Data.totensor(validds,CFG.fieldvars,CFG.localvars,CFG.targetvar)
    nfieldvars,nlocalvars,nlevs = len(CFG.fieldvars),len(CFG.localvars),fieldtrain.shape[3]
    loaderkwargs = dict(num_workers=CFG.workers,pin_memory=(DEVICE=='cuda'),persistent_workers=(CFG.workers>0))
    if CFG.workers>0:
        loaderkwargs['prefetch_factor'] = 2
    logger.info('Training selected models...')
    cache = None
    for modelcfg in CFG.models:
        modelname = modelcfg['name']
        if requested is not None and modelname not in requested:
            continue
        logger.info(f'=== Training model: {modelname} ({modelcfg["type"]}) ===')
        patchcfg,uselocal = modelcfg['patch'],modelcfg.get('uselocal',True)
        config = (patchcfg['radius'],patchcfg['maxlevs'],patchcfg['timelag'],uselocal)
        if cache and cache['config']==config:
            logger.info('   Reusing cached datasets and loaders')
            trainloader,validloader,patchshape = cache['trainloader'],cache['validloader'],cache['patchshape']
        else:
            patch = Patch(patchcfg['radius'],patchcfg['maxlevs'],patchcfg['timelag'])
            patchshape = patch.shape(nlevs)
            centerstrain = patch.centers(targettrain,trainds.lat.values,trainds.lon.values,CFG.latrange,CFG.lonrange)
            centersvalid = patch.centers(targetvalid,validds.lat.values,validds.lon.values,CFG.latrange,CFG.lonrange)
            logger.info(f'   Patch: {patchshape} | Train: {len(centerstrain)} | Valid: {len(centersvalid)}')
            traindataset = SampleDataset(fieldtrain,localtrain,targettrain,centerstrain,patch,uselocal)
            validdataset = SampleDataset(fieldvalid,localvalid,targetvalid,centersvalid,patch,uselocal)
            trainloader = DataLoader(traindataset,batch_size=CFG.batchsize,shuffle=True,**loaderkwargs)
            validloader = DataLoader(validdataset,batch_size=CFG.batchsize,shuffle=False,**loaderkwargs)
            cache = {'config':config,'trainloader':trainloader,'validloader':validloader,'patchshape':patchshape}
        model = build(modelcfg,patchshape,nfieldvars,nlocalvars)
        fit(model,modelname,trainloader,validloader)
    logger.info('Finished training selected models.')