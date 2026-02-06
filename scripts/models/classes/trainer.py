#!/usr/bin/env python

import os
import time
import torch
import wandb
import logging
from torch.amp import autocast,GradScaler

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self,model,trainloader,validloader,device,modeldir,project,seed,lr=1e-3,patience=5,
                 criterion='MSELoss',epochs=100,useamp=True,accumsteps=1,compile=False):
        '''
        Purpose: Initialize Trainer with model, dataloaders, and training configuration.
        Args:
        - model (torch.nn.Module): initialized model instance
        - trainloader (torch.utils.data.DataLoader): training dataloader
        - validloader (torch.utils.data.DataLoader): validation dataloader
        - device (str): device to use (cuda or cpu)
        - modeldir (str): output directory for checkpoints
        - project (str): project name for Weights & Biases logging
        - seed (int): random seed for reproducibility
        - lr (float): initial learning rate (defaults to 1e-3)
        - patience (int): early stopping patience (defaults to 5)
        - criterion (str): loss function name (defaults to MSELoss)
        - epochs (int): maximum number of epochs (defaults to 100)
        - useamp (bool): whether to use automatic mixed precision (defaults to True for CUDA)
        - accumsteps (int): gradient accumulation steps for larger effective batch size (defaults to 1)
        - compile (bool): whether to use torch.compile for faster training (defaults to False)
        '''
        self.model       = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.device      = device
        self.modeldir    = modeldir
        self.project     = project
        self.seed        = seed
        self.lr          = lr
        self.patience    = patience
        self.epochs      = epochs
        self.useamp      = useamp and (device=='cuda')
        self.accumsteps  = accumsteps
        self.optimizer   = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=2,min_lr=1e-6)
        self.scaler      = GradScaler('cuda') if self.useamp else None
        self.criterion   = getattr(torch.nn,criterion)()
        if compile and hasattr(torch,'compile'):
            logger.info('   Compiling model with torch.compile...')
            self.model = torch.compile(self.model)

    def save_checkpoint(self,name,state,kind):
        '''
        Purpose: Save best model checkpoint and verify by reopening.
        Args:
        - name (str): model name
        - state (dict): model state_dict to save
        - kind (str): baseline, nonparametric, or parametric
        Returns:
        - bool: True if save successful, False otherwise
        '''
        os.makedirs(self.modeldir,exist_ok=True)
        filename = f'{name}_{self.seed}.pth'
        filepath = os.path.join(self.modeldir,filename)
        logger.info(f'      Attempting to save {filename}...')
        try:
            torch.save(state,filepath)
            _ = torch.load(filepath,map_location='cpu')
            logger.info('         File write successful')
            return True
        except Exception:
            logger.exception('         Failed to save or verify')
            return False

    def train_epoch(self,uselocal,haskernel):
        '''
        Purpose: Execute one training epoch with gradient accumulation and mixed precision.
        Args:
        - uselocal (bool): whether to use local inputs
        - haskernel (bool): whether model has integration kernel
        Returns:
        - float: average training loss for the epoch
        '''
        self.model.train()
        totalloss = 0.0
        self.optimizer.zero_grad()
        dataloadtime = 0.0
        computetime = 0.0
        batchstart = time.time()
        for idx,batch in enumerate(self.trainloader):
            dataloadtime += time.time() - batchstart
            computestart = time.time()
            fieldpatch   = batch['fieldpatch'].to(self.device,non_blocking=True)
            localvalues  = batch['localvalues'].to(self.device,non_blocking=True) if uselocal else None
            targetvalues = batch['targetvalues'].to(self.device,non_blocking=True)
            if haskernel:
                dareapatch = batch['dareapatch'].to(self.device,non_blocking=True)
                dlevpatch  = batch['dlevpatch'].to(self.device,non_blocking=True)
                dtimepatch = batch['dtimepatch'].to(self.device,non_blocking=True)
                dlevfull   = batch['dlevfull'].to(self.device,non_blocking=True)
            if self.useamp:
                with autocast('cuda',enabled=self.useamp):
                    if haskernel:
                        outputvalues = self.model(fieldpatch,dareapatch,dlevpatch,dtimepatch,dlevfull,localvalues)
                    else:
                        outputvalues = self.model(fieldpatch,localvalues)
                    loss = self.criterion(outputvalues,targetvalues)
                    loss = loss/self.accumsteps
                self.scaler.scale(loss).backward()
                if (idx+1)%self.accumsteps==0 or (idx+1)==len(self.trainloader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                if haskernel:
                    outputvalues = self.model(fieldpatch,dareapatch,dlevpatch,dtimepatch,dlevfull,localvalues)
                else:
                    outputvalues = self.model(fieldpatch,localvalues)
                loss = self.criterion(outputvalues,targetvalues)
                loss = loss/self.accumsteps
                loss.backward()
                if (idx+1)%self.accumsteps==0 or (idx+1)==len(self.trainloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            totalloss += loss.detach()*self.accumsteps*targetvalues.numel()
            computetime += time.time() - computestart
            batchstart = time.time()
        avgloss = (totalloss/len(self.trainloader.dataset)).item()
        return avgloss,dataloadtime,computetime

    def validate_epoch(self,uselocal,haskernel):
        '''
        Purpose: Execute one validation epoch.
        Args:
        - uselocal (bool): whether to use local inputs
        - haskernel (bool): whether model has integration kernel
        Returns:
        - float: average validation loss for the epoch
        '''
        self.model.eval()
        totalloss = 0.0
        with torch.no_grad():
            for batch in self.validloader:
                fieldpatch   = batch['fieldpatch'].to(self.device,non_blocking=True)
                localvalues  = batch['localvalues'].to(self.device,non_blocking=True) if uselocal else None
                targetvalues = batch['targetvalues'].to(self.device,non_blocking=True)
                if haskernel:
                    dareapatch = batch['dareapatch'].to(self.device,non_blocking=True)
                    dlevpatch  = batch['dlevpatch'].to(self.device,non_blocking=True)
                    dtimepatch = batch['dtimepatch'].to(self.device,non_blocking=True)
                    dlevfull   = batch['dlevfull'].to(self.device,non_blocking=True)
                if self.useamp:
                    with autocast('cuda',enabled=self.useamp):
                        if haskernel:
                            outputvalues = self.model(fieldpatch,dareapatch,dlevpatch,dtimepatch,dlevfull,localvalues)
                        else:
                            outputvalues = self.model(fieldpatch,localvalues)
                        loss = self.criterion(outputvalues,targetvalues)
                else:
                    if haskernel:
                        outputvalues = self.model(fieldpatch,dareapatch,dlevpatch,dtimepatch,dlevfull,localvalues)
                    else:
                        outputvalues = self.model(fieldpatch,localvalues)
                    loss = self.criterion(outputvalues,targetvalues)
                totalloss += loss.detach()*targetvalues.numel()
        return (totalloss/len(self.validloader.dataset)).item()

    def fit(self,name,kind,uselocal):
        '''
        Purpose: Train model with early stopping and learning rate scheduling.
        Args:
        - name (str): model name
        - kind (str): baseline, nonparametric, or parametric
        - uselocal (bool): whether to use local inputs
        '''
        haskernel = hasattr(self.model,'kernel')
        trainsamples = len(self.trainloader.dataset)
        validsamples = len(self.validloader.dataset)
        logger.info(f'   Training samples: {trainsamples}, Validation samples: {validsamples}')
        logger.info(f'   Training batches: {len(self.trainloader)}, Validation batches: {len(self.validloader)}')
        wandb.init(project=self.project,name=name,
                   config={
                       'Seed':self.seed,
                       'Epochs':self.epochs,
                       'Effective batch size':self.trainloader.batch_size*self.accumsteps,
                       'Initial learning rate':self.lr,
                       'Early stopping patience':self.patience,
                       'Loss function':self.criterion.__class__.__name__,
                       'Number of parameters':self.model.nparams if hasattr(self.model,'nparams') else sum(p.numel() for p in self.model.parameters()),
                       'Device':self.device,
                       'Mixed precision':self.useamp,
                       'Training samples':trainsamples,
                       'Validation samples':validsamples})
        beststate = None
        bestloss  = float('inf')
        bestepoch = 0
        noimprove = 0
        starttime = time.time()
        for epoch in range(1,self.epochs+1):
            epochstart = time.time()
            trainloss,dataloadtime,computetime = self.train_epoch(uselocal,haskernel)
            epochtime = time.time() - epochstart
            validloss = self.validate_epoch(uselocal,haskernel)
            self.scheduler.step(validloss)
            if validloss<bestloss:
                beststate = {key:value.detach().cpu().clone() for key,value in self.model.state_dict().items()}
                bestloss  = validloss
                bestepoch = epoch
                noimprove = 0
            else:
                noimprove += 1
            datapct = 100.0*dataloadtime/(dataloadtime+computetime)
            computepct = 100.0*computetime/(dataloadtime+computetime)
            wandb.log({
                'Epoch': epoch,
                'Training loss':trainloss,
                'Validation loss':validloss,
                'Learning rate':self.optimizer.param_groups[0]['lr']})
            logger.info(f'   Epoch {epoch}/{self.epochs}: train_loss={trainloss:.6f}, valid_loss={validloss:.6f}, lr={self.optimizer.param_groups[0]["lr"]:.2e}')
            logger.info(f'   Timing: epoch={epochtime/60:.1f}min, data={dataloadtime/60:.1f}min ({datapct:.0f}%), compute={computetime/60:.1f}min ({computepct:.0f}%)')
            if noimprove>=self.patience:
                break
        duration = time.time()-starttime
        wandb.run.summary.update({'Best validation loss':bestloss})
        logger.info(f'   Training complete: duration={duration/60:.1f}min, best_epoch={bestepoch}, best_loss={bestloss:.6f}')
        if beststate is not None:
            self.save_checkpoint(name,beststate,kind)
        wandb.finish()
