#!/usr/bin/env python

import os
import torch
import logging
from utils import Config
from dataset import DataPrep
from network import NonparametricKernelLayer,ParametricKernelLayer,BaselineNN,KernelNN

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
SEED      = config.seed
WORKERS   = config.workers
EPOCHS    = config.epochs
BATCHSIZE = config.batchsize
LR        = config.learningrate
PATIENCE  = config.patience
CRITERION = config.criterion

def build(name,modelconfig,nfieldvars,nlocalvars,patchshape):
    '''
    Purpose: Build a model instance from configuration.
    Args:
    - name (str): model name
    - modelconfig (dict): model configuration
    - nfieldvars (int): number of predictor fields
    - nlocalvars (int): number of local inputs
    - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
    Returns:
    - torch.nn.Module: initialized model
    '''
    modeltype = modelconfig['type']
    uselocal  = modelconfig['uselocal']
    if modeltype=='baseline':
        model = BaselineNN(nfieldvars,patchshape,nlocalvars,uselocal)
    elif modeltype=='nonparametric':
        nkernels     = modelconfig['nkernels']
        kernelconfig = modelconfig['kernelconfig']
        kernellayer  = NonparametricKernelLayer(nfieldvars,patchshape,nkernels,kernelconfig)
        model = KernelNN(kernellayer,nlocalvars,uselocal)
    elif modeltype=='parametric':
        nkernels     = modelconfig['nkernels']
        kernelconfig = modelconfig['kernelconfig']
        kernellayer  = ParametricKernelLayer(nfieldvars,patchshape,nkernels,kernelconfig)
        model = KernelNN(kernellayer,nlocalvars,uselocal)
    else:
        raise ValueError(f'Unknown model type `{modeltype}`')
    logger.info(f'   Built {name} model with {sum(p.numel() for p in model.parameters())} parameters')
    return model



def train_model(modelname,model,trainloader,validloader,device,uselocal,quadweights,
                epochs=EPOCHS,lr=LR,patience=PATIENCE,criterion=CRITERION,modeldir=MODELDIR):
    '''
    Purpose: Train a model with early stopping and save the best checkpoint.
    Args:
    - modelname (str): name of the model
    - model (torch.nn.Module): model to train
    - trainloader (DataLoader): training data loader
    - validloader (DataLoader): validation data loader
    - device (str): device to use
    - uselocal (bool): whether to use local inputs
    - quadweights (torch.Tensor | None): quadrature weights for kernel models
    - epochs (int): maximum number of epochs (defaults to EPOCHS)
    - lr (float): learning rate (defaults to LR)
    - patience (int): early stopping patience (defaults to PATIENCE)
    - criterion (str): loss function name (defaults to CRITERION)
    - modeldir (str): directory to save model checkpoints (defaults to MODELDIR)
    Returns:
    - torch.nn.Module: best model
    '''
    os.makedirs(modeldir,exist_ok=True)
    if criterion=='mse':
        criterion_fn = torch.nn.MSELoss()
    elif criterion=='mae':
        criterion_fn = torch.nn.L1Loss()
    else:
        raise ValueError(f'Unknown criterion: {criterion}')
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    bestloss = float('inf')
    counter  = 0
    for epoch in range(epochs):
        model.train()
        totalloss = 0.0
        for batch in loader:
            patch  = batch['patch'].to(device)
            target = batch['target'].to(device)
            local  = batch.get('local').to(device) if uselocal else None
            optimizer.zero_grad()
            if quadweights is not None:
                output = model(patch,local,quadweights.to(device))
            else:
                output = model(patch,local)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            totalloss += loss.item()*len(target)
        trainloss =  totalloss/len(loader.dataset)
        model.eval()
        totalloss = 0.0
        with torch.no_grad():
            for batch in loader:
                patch  = batch['patch'].to(device)
                target = batch['target'].to(device)
                local  = batch.get('local').to(device) if uselocal else None
                if quadweights is not None:
                    output = model(patch,local,quadweights.to(device))
                else:
                    output = model(patch,local)
                loss = criterion(output,target)
                totalloss += loss.item()*len(target)
        validloss = totalloss/len(loader.dataset)
        logger.info(f'   Epoch {epoch+1:2d}/{epochs}: train={trainloss:.6f}, valid={validloss:.6f}')
        if validloss<bestloss:
            bestloss = validloss
            counter  = 0
            filename = f'{name}.pth'
            filepath = os.path.join(modeldir,filename)
            torch.save(model.state_dict(),filepath)
            logger.info(f'      Saved best model to {filename}')
        else:
            counter += 1
            if counter>=patience:
                logger.info(f'      Early stopping at epoch {epoch+1}')
                break
    model.load_state_dict(torch.load(os.path.join(modeldir,f'{modelname}.pt')))
    return model


def save(modelstate,name,modeldir=MODELDIR):
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
    filename = f'{name}.pth'
    filepath = os.path.join(modeldir, filename)
    logger.info(f'   Attempting to save {filename}...')
    try:
        torch.save(modelstate,filepath)
        _ = torch.load(filepath,map_location='cpu')
        logger.info('      File write successful')
        return True
    except Exception:
        logger.exception('      Failed to save or verify')
        return False

if __name__=='__main__':
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Training selected NN models on {device}')
    logger.info('Preparing data splits...')
    splitdata  = DataPrep.prepare(SPLITDIR,FIELDVARS,LOCALVARS,TARGETVAR,['train','valid'])
    for name,modelconfig in config.models.items():
        logger.info(f'Training {model}...')
        patchconfig = modelconfig['patchconfig']
        uselocal    = modelconfig['uselocal']
        data  = DataPrep.dataloaders(splitdata,patchconfig,uselocal,LATRANGE,LONRANGE,BATCHSIZE,WORKERS,device,splitdata['train']['field'].shape[3])
        model = build(name,modelconfig,len(FIELDVARS),len(LOCALVARS),data['patchshape']).to(device)
        qweights = data['quadweights'] if hasattr(model,'kernellayer') else None
        model = fit(name,model,data['loaders']['train'],data['loaders']['valid'],device,uselocal,quadweights)
        logger.info(f'Finished training {modelname}')