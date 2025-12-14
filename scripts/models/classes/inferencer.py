#!/usr/bin/env python

import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Inferencer:

    def __init__(self,model,dataloader,device):
        '''
        Purpose: Initialize Inferencer for model predictions on test/validation data.
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
        - np.ndarray: predictions array with shape (nsamples,) or (nsamples, nkernels)
        '''
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in self.dataloader:
                fieldpatch   = batch['fieldpatch'].to(self.device,non_blocking=True)
                localvalues  = batch['localvalues'].to(self.device,non_blocking=True) if uselocal else None
                if haskernel:
                    dareapatch = batch['dareapatch'].to(self.device,non_blocking=True)
                    dlevpatch  = batch['dlevpatch'].to(self.device,non_blocking=True)
                    dtimepatch = batch['dtimepatch'].to(self.device,non_blocking=True)
                    outputvalues = self.model(fieldpatch,dareapatch,dlevpatch,dtimepatch,localvalues)
                    del fieldpatch,localvalues,dareapatch,dlevpatch,dtimepatch
                else:
                    outputvalues = self.model(fieldpatch,localvalues)
                    del fieldpatch,localvalues
                predictions.append(outputvalues.cpu().numpy())
                del outputvalues
        return np.concatenate(predictions,axis=0)


    def extract_features(self,uselocal):
        '''
        Purpose: Extract kernel features for all samples in the dataloader.
        Args:
        - uselocal (bool): whether to use local inputs
        Returns:
        - np.ndarray: features array with shape (nsamples, nfeatures)
        '''
        if not hasattr(self.model,'intkernel'):
            raise ValueError('Model must have integration kernel to extract features')
        self.model.eval()
        features = []
        with torch.no_grad():
            for batch in self.dataloader:
                fieldpatch = batch['fieldpatch'].to(self.device,non_blocking=True)
                dareapatch = batch['dareapatch'].to(self.device,non_blocking=True)
                dlevpatch  = batch['dlevpatch'].to(self.device,non_blocking=True)
                dtimepatch = batch['dtimepatch'].to(self.device,non_blocking=True)
                _ = self.model.intkernel(fieldpatch,dareapatch,dlevpatch,dtimepatch)  
                featureten = self.model.intkernel.features                              
                features.append(featureten.cpu().numpy())
        return np.concatenate(features,axis=0)

    def extract_weights(self):
        '''
        Purpose: Extract kernel weights from the model.
        Returns:
        - np.ndarray: weights array
        '''
        if not hasattr(self.model,'intkernel'):
            raise ValueError('Model must have integration kernel to extract weights')
        self.model.eval()
        with torch.no_grad():
            weights = self.model.intkernel.weights.cpu().numpy()
        return weights
