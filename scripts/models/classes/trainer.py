#!/usr/bin/env python

import os
import time
import torch
import wandb
import logging
from torch.amp import autocast,GradScaler

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self,model,trainloader,validloader,device,modeldir,project,lr=1e-3,patience=5,
                 criterion='MSELoss',epochs=100,use_amp=True,grad_accum_steps=1,compile_model=False):
        '''
        Purpose: Initialize Trainer with model, dataloaders, and training configuration.
        Args:
        - model (torch.nn.Module): initialized model instance
        - trainloader (torch.utils.data.DataLoader): training dataloader
        - validloader (torch.utils.data.DataLoader): validation dataloader
        - device (str): device to use (cuda or cpu)
        - modeldir (str): output directory for checkpoints
        - project (str): project name for Weights & Biases logging
        - lr (float): initial learning rate (defaults to 1e-3)
        - patience (int): early stopping patience (defaults to 5)
        - criterion (str): loss function name (defaults to MSELoss)
        - epochs (int): maximum number of epochs (defaults to 100)
        - use_amp (bool): whether to use automatic mixed precision (defaults to True for CUDA)
        - grad_accum_steps (int): gradient accumulation steps for larger effective batch size (defaults to 1)
        - compile_model (bool): whether to use torch.compile for faster training (defaults to False)
        '''
        self.model          = model
        self.trainloader    = trainloader
        self.validloader    = validloader
        self.device         = device
        self.modeldir       = modeldir
        self.project        = project
        self.lr             = lr
        self.patience       = patience
        self.epochs         = epochs
        self.use_amp        = use_amp and (device=='cuda')
        self.grad_accum_steps = grad_accum_steps
        self.optimizer      = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.scheduler      = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=2,min_lr=1e-6)
        self.scaler         = GradScaler('cuda') if self.use_amp else None
        self.criterion      = getattr(torch.nn,criterion)()
        if compile_model and hasattr(torch,'compile'):
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
        savedir = os.path.join(self.modeldir,kind)
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
        for idx,batch in enumerate(self.trainloader):
            fieldpatch   = batch['fieldpatch'].to(self.device,non_blocking=True)
            localvalues  = batch['localvalues'].to(self.device,non_blocking=True) if uselocal else None
            targetvalues = batch['targetvalues'].to(self.device,non_blocking=True)
            if haskernel:
                dareapatch = batch['dareapatch'].to(self.device,non_blocking=True)
                dlevpatch  = batch['dlevpatch'].to(self.device,non_blocking=True)
                dtimepatch = batch['dtimepatch'].to(self.device,non_blocking=True)
            if self.use_amp:
                with autocast('cuda',enabled=self.use_amp):
                    if haskernel:
                        outputvalues = self.model(fieldpatch,dareapatch,dlevpatch,dtimepatch,localvalues)
                    else:
                        outputvalues = self.model(fieldpatch,localvalues)
                    loss = self.criterion(outputvalues,targetvalues)
                    loss = loss/self.grad_accum_steps
                self.scaler.scale(loss).backward()
                if (idx+1)%self.grad_accum_steps==0 or (idx+1)==len(self.trainloader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                if haskernel:
                    outputvalues = self.model(fieldpatch,dareapatch,dlevpatch,dtimepatch,localvalues)
                else:
                    outputvalues = self.model(fieldpatch,localvalues)
                loss = self.criterion(outputvalues,targetvalues)
                loss = loss/self.grad_accum_steps
                loss.backward()
                if (idx+1)%self.grad_accum_steps==0 or (idx+1)==len(self.trainloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            totalloss += loss.detach()*self.grad_accum_steps*targetvalues.numel()
        return (totalloss/len(self.trainloader.dataset)).item()

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
                if self.use_amp:
                    with autocast('cuda',enabled=self.use_amp):
                        if haskernel:
                            outputvalues = self.model(fieldpatch,dareapatch,dlevpatch,dtimepatch,localvalues)
                        else:
                            outputvalues = self.model(fieldpatch,localvalues)
                        loss = self.criterion(outputvalues,targetvalues)
                else:
                    if haskernel:
                        outputvalues = self.model(fieldpatch,dareapatch,dlevpatch,dtimepatch,localvalues)
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
        haskernel = hasattr(self.model,'intkernel')
        wandb.init(project=self.project,name=name,
                   config={
                       'Epochs':self.epochs,
                       'Batch size':self.trainloader.batch_size,
                       'Effective batch size':self.trainloader.batch_size*self.grad_accum_steps,
                       'Initial learning rate':self.lr,
                       'Early stopping patience':self.patience,
                       'Loss function':self.criterion.__class__.__name__,
                       'Number of parameters':self.model.nparams if hasattr(self.model,'nparams') else sum(p.numel() for p in self.model.parameters()),
                       'Device':self.device,
                       'Mixed precision':self.use_amp,
                       'Gradient accumulation steps':self.grad_accum_steps})
        beststate = None
        bestloss  = float('inf')
        bestepoch = 0
        noimprove = 0
        starttime = time.time()
        for epoch in range(1,self.epochs+1):
            trainloss = self.train_epoch(uselocal,haskernel)
            validloss = self.validate_epoch(uselocal,haskernel)
            self.scheduler.step(validloss)
            if validloss<bestloss:
                beststate = {key:value.detach().cpu().clone() for key,value in self.model.state_dict().items()}
                bestloss  = validloss
                bestepoch = epoch
                noimprove = 0
            else:
                noimprove += 1
            wandb.log({
                'Epoch':epoch,
                'Training loss':trainloss,
                'Validation loss':validloss,
                'Learning rate':self.optimizer.param_groups[0]['lr']})
            if noimprove>=self.patience:
                break
        duration = time.time()-starttime
        wandb.run.summary.update({
            'Best model at epoch':bestepoch,
            'Best validation loss':bestloss,
            'Training duration (s)':duration,
            'Stopped early':noimprove>=self.patience})
        if beststate is not None:
            self.save_checkpoint(name,beststate,kind)
        wandb.finish()
