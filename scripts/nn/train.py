#!/usr/bin/env python

import os
import time
import argparse
import logging
import warnings

import numpy as np
import torch
import wandb
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

CFG         = Config()
FILEDIR     = CFG.filedir
MODELDIR    = CFG.modeldir
FIELDVARS   = CFG.fieldvars
LOCALVARS   = CFG.localvars
TARGETVAR   = CFG.targetvar
LATRANGE    = CFG.latrange
LONRANGE    = CFG.lonrange
PATCH_CFG   = CFG.patch
TRAIN_CFG   = CFG.training
MODEL_CFGS  = CFG.models

SEED         = CFG.seed
WORKERS      = CFG.workers
EPOCHS       = CFG.epochs
BATCHSIZE    = CFG.batchsize
LEARNINGRATE = CFG.learningrate
PATIENCE     = CFG.patience
CRIT_NAME    = CFG.criterion

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP_ENABLED = (DEVICE=='cuda')
if DEVICE=='cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_criterion(name):
    '''
    Purpose: Build a torch loss function from a string name in configs.json.
    '''
    if hasattr(torch.nn,name):
        return getattr(torch.nn,name)()
    raise ValueError(f'Unknown loss function `{name}` in configs.json')
    

CRITERION = get_criterion(CRIT_NAME)


# -----------------------------------------------------------------------------#
# Training utilities
# -----------------------------------------------------------------------------#

def save_best(modelstate,runname,modeldir=MODELDIR):
    '''
    Purpose: Save best (lowest validation loss) checkpoint for a run.
    '''
    os.makedirs(modeldir,exist_ok=True)
    filename = f'{runname}_best.pth'
    filepath = os.path.join(modeldir,filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        torch.save(modelstate,filepath)
        _ = torch.load(filepath,map_location='cpu')
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False


def fit(model,runname,trainloader,validloader,
        device=DEVICE,learningrate=LEARNINGRATE,
        patience=PATIENCE,criterion=CRITERION,epochs=EPOCHS):
    '''
    Purpose: Train a NN with early stopping and OneCycleLR, log to W&B, and save only the best checkpoint.
    '''
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learningrate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,max_lr=learningrate,
        epochs=epochs,steps_per_epoch=len(trainloader),
        pct_start=0.1,anneal_strategy='cos')
    scaler    = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    wandb.init(
        project='Chapter 2 Experiments',
        name=runname,
        config={
            'Epochs':epochs,
            'Batch size':trainloader.batch_size,
            'Initial learning rate':learningrate,
            'Early stopping patience':patience,
            'Loss function':CRIT_NAME,
            'AMP enabled':AMP_ENABLED})

    bestloss  = float('inf')
    beststate = None
    bestepoch = 0
    noimprove = 0
    starttime = time.time()

    for epoch in range(1,epochs+1):
        # --- train ---
        model.train()
        runningloss = 0.0
        for batch in trainloader:
            patch = batch['patch'].to(device,non_blocking=True)
            local = batch.get('local',None)
            if local is not None:
                local = local.to(device,non_blocking=True)
            target = batch['target'].to(device,non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                ypred = model(patch,local).squeeze(-1)
                loss  = criterion(ypred,target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            runningloss += loss.item()*patch.size(0)
        trainloss = runningloss/len(trainloader.dataset)

        # --- validate ---
        model.eval()
        runningloss = 0.0
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                for batch in validloader:
                    patch = batch['patch'].to(device,non_blocking=True)
                    local = batch.get('local',None)
                    if local is not None:
                        local = local.to(device,non_blocking=True)
                    target = batch['target'].to(device,non_blocking=True)
                    ypred  = model(patch,local).squeeze(-1)
                    loss   = criterion(ypred,target)
                    runningloss += loss.item()*patch.size(0)
        validloss = runningloss/len(validloader.dataset)

        wandb.log({
            'Epoch':epoch,
            'Training loss':trainloss,
            'Validation loss':validloss,
            'Learning rate':optimizer.param_groups[0]['lr']})

        logger.info(f'Epoch {epoch:03d} | Train: {trainloss:.6e} | Valid: {validloss:.6e}')

        if validloss<bestloss:
            bestloss  = validloss
            bestepoch = epoch
            noimprove = 0
            beststate = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            noimprove += 1

        if noimprove>=patience:
            logger.info(f'Early stopping at epoch {epoch:03d}')
            break

    duration = time.time()-starttime
    wandb.run.summary.update({
        'Best model epoch':bestepoch,
        'Best validation loss':bestloss,
        'Total epochs':epoch,
        'Training duration (s)':duration,
        'Stopped early':noimprove>=patience})
    wandb.finish()

    if beststate is not None:
        save_best(beststate,runname)


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#

def parse_args():
    parser = argparse.ArgumentParser(description='Train NN precipitation models.')
    parser.add_argument(
        '--models',
        type=str,
        default='all',
        help='Comma-separated list of model names to train (as in configs.json "models.name"), or "all".')
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    requested = [m.strip() for m in args.models.split(',')] if args.models!='all' else None

    logger.info('Loading training and validation splits...')
    trainds = load('train',FILEDIR)
    validds = load('valid',FILEDIR)

    logger.info('Converting splits to torch tensors...')
    fieldtrain,localtrain,targettrain = tensors(trainds,FIELDVARS,LOCALVARS,TARGETVAR)
    fieldvalid,localvalid,targetvalid = tensors(validds,FIELDVARS,LOCALVARS,TARGETVAR)

    logger.info('Configuring patch...')
    patch = Patch(
        latradius=PATCH_CFG['latradius'],
        lonradius=PATCH_CFG['lonradius'],
        maxlevs=PATCH_CFG['maxlevs'],
        timelag=PATCH_CFG['timelag'])
    nlevs_full = fieldtrain.shape[3]
    patchshape = patch.shape(nlevs_full)

    logger.info('Building sample centers...')
    centerstrain = patch.centers(
        targetdata=targettrain,
        lats=trainds['lat'].values,
        lons=trainds['lon'].values,
        latrange=LATRANGE,
        lonrange=LONRANGE)
    centersvalid = patch.centers(
        targetdata=targetvalid,
        lats=validds['lat'].values,
        lons=validds['lon'].values,
        latrange=LATRANGE,
        lonrange=LONRANGE)

    logger.info(f'Training samples: {len(centerstrain)}, validation samples: {len(centersvalid)}')

    logger.info('Building SampleDataset objects...')
    traindataset = SampleDataset(
        fielddata=fieldtrain,
        localdata=localtrain,
        targetdata=targettrain,
        centers=centerstrain,
        patch=patch)
    validdataset = SampleDataset(
        fielddata=fieldvalid,
        localdata=localvalid,
        targetdata=targetvalid,
        centers=centersvalid,
        patch=patch)

    logger.info('Building DataLoaders...')
    common_loader_kwargs = dict(
        num_workers=WORKERS,
        pin_memory=(DEVICE=='cuda'),
        persistent_workers=(WORKERS>0))
    if WORKERS>0:
        common_loader_kwargs['prefetch_factor'] = 2

    trainloader = DataLoader(
        traindataset,batch_size=BATCHSIZE,shuffle=True,
        **common_loader_kwargs)
    validloader = DataLoader(
        validdataset,batch_size=BATCHSIZE,shuffle=False,
        **common_loader_kwargs)

    nfieldvars = len(FIELDVARS)
    nlocalvars = len(LOCALVARS)

    logger.info('Training selected models...')
    for modelcfg in MODEL_CFGS:
        modelname = modelcfg['name']
        if (requested is not None) and (modelname not in requested):
            continue
        logger.info(f'=== Training model: {modelname} ({modelcfg["type"]}) ===')
        model = build_model_from_config(modelcfg,patchshape,nfieldvars,nlocalvars)
        fit(model,modelname,trainloader,validloader)

    logger.info('Finished training selected models.')