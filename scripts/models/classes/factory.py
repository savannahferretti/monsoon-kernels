#!/usr/bin/env python

from scripts.models.architectures import MainNN,BaselineNN,KernelNN
from scripts.models.kernels import NonparametricKernelLayer,ParametricKernelLayer

class ModelFactory:

    @staticmethod
    def build(name,modelconfig,patchshape,nfieldvars,nlocalvars):
        '''
        Purpose: Build a model instance from configuration.
        Args:
        - name (str): model name
        - modelconfig (dict[str,object]): model configuration
        - patchshape (tuple[int,int,int,int]): (plats, plons, plevs, ptimes)
        - nfieldvars (int): number of predictor fields
        - nlocalvars (int): number of local inputs
        Returns:
        - torch.nn.Module: initialized model
        '''
        kind = modelconfig['kind']
        uselocal = modelconfig['uselocal']
        if kind=='baseline':
            model = BaselineNN(patchshape,nfieldvars,nlocalvars,uselocal)
        elif kind=='nonparametric':
            kerneldims = modelconfig['kerneldims']
            kernel     = NonparametricKernelLayer(nfieldvars,kerneldims,patchshape)
            model      = KernelNN(kernel,nlocalvars,uselocal,patchshape)
        elif kind=='parametric':
            kerneldict = modelconfig['kerneldict']
            kernel     = ParametricKernelLayer(nfieldvars,kerneldict)
            model      = KernelNN(kernel,nlocalvars,uselocal,patchshape)
        else:
            raise ValueError(f'Unknown model kind `{kind}`')
        model.nparams = sum(param.numel() for param in model.parameters())
        return model
