#!/usr/bin/env python

import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Inferencer:

    def __init__(self,model,dataloader,device):
        '''
        Purpose: Initialize Inferencer for model evaluation on test/validation data.
        Args:
        - model (torch.nn.Module): trained model instance
        - dataloader (torch.utils.data.DataLoader): dataloader for inference
        - device (str): device to use (cuda or cpu)
        '''
        self.model      = model
        self.dataloader = dataloader
        self.device     = device

    def predict(self,uselocal,haskernel):
        '''
        Purpose: Generate predictions for all samples in the dataloader.
        Args:
        - uselocal (bool): whether to use local inputs
        - haskernel (bool): whether model has integration kernel
        Returns:
        - np.ndarray: predictions array with shape (nsamples,)
        '''
        self.model.eval()
        predslist = []
        with torch.no_grad():
            for batch in self.dataloader:
                fieldpatch  = batch['fieldpatch'].to(self.device,non_blocking=True)
                localvalues = batch['localvalues'].to(self.device,non_blocking=True) if uselocal else None
                if haskernel:
                    dareapatch = batch['dareapatch'].to(self.device,non_blocking=True)
                    dlevpatch  = batch['dlevpatch'].to(self.device,non_blocking=True)
                    dtimepatch = batch['dtimepatch'].to(self.device,non_blocking=True)
                    dlevfull   = batch['dlevfull'].to(self.device,non_blocking=True)
                    output     = self.model(fieldpatch,dareapatch,dlevpatch,dtimepatch,dlevfull,localvalues)
                else:
                    output     = self.model(fieldpatch,localvalues)
                predslist.append(output.detach().cpu().numpy())
        return np.concatenate(predslist,axis=0).astype(np.float32)

    def extract_weights(self,nonparam):
        '''
        Purpose: Extract normalized kernel weights and optional mixture component weights.
        Args:
        - nonparam (bool): whether the kernel is non-parametric
        Returns:
        - tuple[np.ndarray, np.ndarray | None]: weights array and component weights (or None)
        '''
        if self.model.intkernel.weights is None:
            raise RuntimeError('`model.intkernel.weights` was not populated during forward pass')
        weights    = self.model.intkernel.weights.detach().cpu().numpy().astype(np.float32)
        components = None
        if not nonparam and hasattr(self.model.intkernel,'get_weights'):
            batch = next(iter(self.dataloader))
            with torch.no_grad():
                dareapatch = batch['dareapatch'].to(self.device,non_blocking=True)
                dlevfull   = batch['dlevfull'].to(self.device,non_blocking=True)
                dtimepatch = batch['dtimepatch'].to(self.device,non_blocking=True)
                self.model.intkernel.get_weights(dareapatch,dlevfull,dtimepatch,self.device,compute_components=True)
            if self.model.intkernel.componentweights is not None:
                components = self.model.intkernel.componentweights.detach().cpu().numpy().astype(np.float32)
        return weights,components
