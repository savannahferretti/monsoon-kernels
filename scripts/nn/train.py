#!/usr/bin/env python

import os
import time
import logging
import warnings
import numpy as np
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

CFG        = Config()
FILEDIR    = CFG.filedir
MODELDIR   = CFG.modeldir
FIELDVARS  = CFG.fieldvars
LOCALVARS  = CFG.localvars
TARGETVAR  = CFG.targetvar
LATRANGE   = CFG.latrange
LONRANGE   = CFG.lonrange
MODEL_CFGS = CFG.models
SEED       = CFG.seed
WORKERS    = CFG.workers
EPOCHS     = CFG.epochs
BATCHSIZE  = CFG.batchsize
LRATE      = CFG.learningrate
PATIENCE   = CFG.patience
CRIT_NAME  = CFG.criterion

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
    Args:
    - name (str): loss function name (e.g., 'MSELoss')
    Returns:
    - torch.nn.Module: loss function instance
    '''
    if hasattr(torch.nn,name):
        return getattr(torch.nn,name)()
    raise ValueError(f'Unknown loss function `{name}` in configs.json')

CRITERION = get_criterion(CRIT_NAME)

def save_best(modelstate,runname,modeldir=MODELDIR):
    '''
    Purpose: Save best (lowest validation loss) checkpoint for a run.
    Args:
    - modelstate (dict): model state dictionary
    - runname (str): model run name
    - modeldir (str): directory to save checkpoints (defaults to MODELDIR)
    Returns:
    - bool: True if save successful, False otherwise
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

def fit(model,runname,trainloader,validloader,device=DEVICE,lrate=LRATE,patience=PATIENCE,criterion=CRITERION,epochs=EPOCHS):
    '''
    Purpose: Train a NN with early stopping and OneCycleLR, and save only the best checkpoint.
    Args:
    - model (torch.nn.Module): model to train
    - runname (str): model run name for saving checkpoints
    - trainloader (DataLoader): training data loader
    - validloader (DataLoader): validation data loader
    - device (str): device to train on (defaults to DEVICE)
    - lrate (float): initial learning rate (defaults to LRATE)
    - patience (int): early stopping patience (defaults to PATIENCE)
    - criterion (torch.nn.Module): loss function (defaults to CRITERION)
    - epochs (int): maximum number of epochs (defaults to EPOCHS)
    '''
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lrate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,max_lr=lrate,
        epochs=epochs,steps_per_epoch=len(trainloader),
        pct_start=0.1,anneal_strategy='cos')
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
    bestloss  = float('inf')
    beststate = None
    bestepoch = 0
    noimprove = 0
    starttime = time.time()
    for epoch in range(1,epochs+1):
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
    logger.info(f'Best epoch: {bestepoch:03d} | Best valid loss: {bestloss:.6e} | Duration: {duration:.1f} s')
    if beststate is not None:
        save_best(beststate,runname)

def parse_args():
    '''
    Purpose: Parse command-line arguments for training script.
    Returns:
    - argparse.Namespace: parsed arguments
    '''
    import argparse
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
    nfieldvars = len(FIELDVARS)
    nlocalvars = len(LOCALVARS)
    nlevs = fieldtrain.shape[3]
    commonkwargs = dict(
        num_workers=WORKERS,
        pin_memory=(DEVICE=='cuda'),
        persistent_workers=(WORKERS>0))
    if WORKERS>0:
        commonkwargs['prefetch_factor'] = 2
    logger.info('Training selected models...')
    cachedconfig = None
    cachedpatch = None
    cachedcenters = None
    cacheddatasets = None
    cachedloaders = None
    for modelcfg in MODEL_CFGS:
        modelname = modelcfg['name']
        if (requested is not None) and (modelname not in requested):
            continue
        logger.info(f'=== Training model: {modelname} ({modelcfg["type"]}) ===')
        patchcfg = modelcfg['patch']
        uselocal = modelcfg.get('uselocal',True)
        currentconfig = (patchcfg['radius'],patchcfg['maxlevs'],patchcfg['timelag'],uselocal)
        if currentconfig==cachedconfig:
            logger.info('   Reusing cached patch, centers, datasets, and loaders')
            patch = cachedpatch
            patchshape = patch.shape(nlevs)
            centerstrain = cachedcenters['train']
            centersvalid = cachedcenters['valid']
            traindataset = cacheddatasets['train']
            validdataset = cacheddatasets['valid']
            trainloader = cachedloaders['train']
            validloader = cachedloaders['valid']
        else:
            patch = Patch(
                radius=patchcfg['radius'],
                maxlevs=patchcfg['maxlevs'],
                timelag=patchcfg['timelag'])
            patchshape = patch.shape(nlevs)
            logger.info(f'   Patch shape: {patchshape}')
            centerstrain = patch.centers(
                targetdata=targettrain,
                lats=trainds.lat.values,
                lons=trainds.lon.values,
                latrange=LATRANGE,
                lonrange=LONRANGE)
            centersvalid = patch.centers(
                targetdata=targetvalid,
                lats=validds.lat.values,
                lons=validds.lon.values,
                latrange=LATRANGE,
                lonrange=LONRANGE)
            logger.info(f'   Train samples: {len(centerstrain)}, valid samples: {len(centersvalid)}')
            traindataset = SampleDataset(
                fielddata=fieldtrain,
                localdata=localtrain,
                targetdata=targettrain,
                centers=centerstrain,
                patch=patch,
                uselocal=uselocal)
            validdataset = SampleDataset(
                fielddata=fieldvalid,
                localdata=localvalid,
                targetdata=targetvalid,
                centers=centersvalid,
                patch=patch,
                uselocal=uselocal)
            trainloader = DataLoader(traindataset,batch_size=BATCHSIZE,shuffle=True,**commonkwargs)
            validloader = DataLoader(validdataset,batch_size=BATCHSIZE,shuffle=False,**commonkwargs)
            cachedconfig = currentconfig
            cachedpatch = patch
            cachedcenters = {'train':centerstrain,'valid':centersvalid}
            cacheddatasets = {'train':traindataset,'valid':validdataset}
            cachedloaders = {'train':trainloader,'valid':validloader}
        model = build_model_from_config(modelcfg,patchshape,nfieldvars,nlocalvars)
        fit(model,modelname,trainloader,validloader)
    logger.info('Finished training selected models.')