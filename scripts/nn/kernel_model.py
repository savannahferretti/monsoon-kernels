#!/usr/bin/env python

import torch

class NNModel(torch.nn.Module):
    
    def __init__(self,inputsize):
        '''
        Purpose: Define a feedforward neural network (NN) for precipitation prediction.
        Args:
        - inputsize (int): number of input features per sample (for 3D variables it's 1 per variable; for 4D variables it's
          'nlevels' per variable; for experiments with more than one variable it's the sum across variables)
        '''
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(inputsize,256), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256,128),       torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128,64),        torch.nn.GELU(), torch.nn.Dropout(0.1),            
            torch.nn.Linear(64,32),         torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(32,1))
    
    def forward(self,X):
        '''
        Purpose: Forward pass through the NN.
        Args:
        - X (torch.Tensor): input features tensor of shape (nsamples, inputsize)
        Returns:
        - torch.Tensor: raw prediction tensor of shape (nsamples, 1)
        '''
        return self.layers(X)

class VerticalKernelLayer(torch.nn.Module):

    def __init__(self,nlevels):
        super().__init__()
        '''
        Purpose: Define a non-parametric vertical integration kernel with one learned weight per level.
        Args:
        - nlevels (int): number of vertical levels in the 4D variable profile
        '''
        self.weights = torch.nn.Parameter(torch.zeros(nlevels))

    def forward(self,x):
        '''
        Purpose: Vertically integrate a 4D variable using the learned kernel (normalized vertical weights).
        Args:
        - x (torch.Tensor): input tensor of shape (nsamples, nlevels)
        Returns:
        - torch.Tensor: integrated input tensor of shape (nsamples, 1)
        '''
        normweights = torch.softmax(self.weights,dim=0).unsqueeze(0)  
        weighted    = x*normweights
        integrated  = weighted.sum(dim=1,keepdim=True)
        return integrated

class GaussianVerticalKernelLayer(torch.nn.Module):

    def __init__(self,levels):
        super().__init__()
        '''
        Purpose: Define a parametric Gaussian vertical integration kernel. The vertical weights
                 are given by a Gaussian function of normalized pressure, with trainable center (mu)
                 and width (sigma). The resulting kernel is normalized to sum to 1.
        Args:
        - levels (np.ndarray): 1D NumPy array of pressure levels (hPa)
        '''
        levels     = torch.as_tensor(levels,dtype=torch.float32)
        normlevels = (levels-levels.min())/(levels.max()-levels.min())
        self.register_buffer('normlevels',normlevels)
        self.mu       = torch.nn.Parameter(torch.tensor(0.5))    
        self.logsigma = torch.nn.Parameter(torch.tensor(-1.0))   

    def forward(self,x):
        '''
        Purpose: Vertically integrate a 4D variable using a Gaussian-shaped learned kernel that is
                 normalized across levels.
        Args:
        - x (torch.Tensor): input tensor of shape (nsamples, nlevels)
        Returns:
        - torch.Tensor: integrated input tensor of shape (nsamples, 1)
        '''
        normlevels = self.normlevels.unsqueeze(0)
        sigma = torch.exp(self.logsigma)      
        gaussweights     = torch.exp(-0.5*((normlevels-self.mu)/sigma)**2)
        normgaussweights = gaussweights/(gaussweights.sum(dim=1,keepdim=True)+1e-12)
        weighted   = x*normgaussweights                    
        integrated = weighted.sum(dim=1,keepdim=True) 
        return integrated

class KernelNNModel(torch.nn.Module):

    def __init__(self,nscalarfeatures,nlevels,kerneltype='vertical',levels=None):
        super().__init__()
        '''
        Purpose: Wrap a feedforward NN (NNModel) with a learned vertical integration kernel for a single 
        4D input variable. The last 'nlevels' columns of the input are treated as the vertical profile; 
        the preceding columns are scalar features. The kernel compresses the profile to one scalar, 
        which is concatenated with the scalar features and passed into the NN.
        Args:
        - nscalarfeatures (int): number of scalar input features (3D variables) per sample
        - nlevels (int): number of vertical levels in the 4D variable profile
        - kerneltype (str): 'vertical' | 'gaussian'
        - levels (np.ndarray or None): 1D array of pressure levels (hPa); required if kerneltype is 'gaussian', otherwise ignored
        '''
        self.nscalarfeatures = nscalarfeatures
        self.nlevels         = nlevels
        if kerneltype=='vertical':
            self.kernel = VerticalKernelLayer(nlevels)
        elif kerneltype=='gaussian':
            if levels is None:
                raise ValueError('GaussianVerticalKernelLayer requires `levels` to be provided.')
            self.kernel = GaussianVerticalKernelLayer(levels)
        else:
            raise ValueError(f"Unknown kerneltype '{kerneltype}', expected 'vertical' or 'gaussian'.")
        inputsize  = nscalarfeatures+1
        self.model = NNModel(inputsize)
       
    def forward(self,X):
        '''
        Purpose: Split the input into scalar features and vertical profiles, apply the vertical kernel to 
                 compress the profile to one scalar, then concatenate and pass through the main NN.
        Args:
        - X (torch.Tensor): input tensor of shape (nsamples, nscalarfeatures+nlevels)
        Returns:
        - torch.Tensor: raw prediction tensor of shape (nsamples, 1)
        '''
        X3D = X[:,:self.nscalarfeatures]
        X4D = X[:,self.nscalarfeatures:]
        inputs = torch.cat([X3D,self.kernel(X4D)],dim=1)
        ypred  = self.model(inputs)
        return ypred