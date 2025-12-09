#!/usr/bin/env python

import os
import torch
import numpy as np
from utils import Config
from io import IO
from data import Data,Patch,SampleDataset
from models import build
from torch.utils.data import DataLoader

def extract(modelname,savedir,extract_features=True):
    '''
    Purpose: Extract and save kernel weights and optionally features from a trained model.
    Args:
    - modelname (str): model name
    - savedir (str): directory to save extracted data
    - extract_features (bool): whether to extract features from training data
    '''
    CFG = Config()
    modelcfg = next(m for m in CFG.models if m['name']==modelname)
    if modelcfg['type']=='baseline':
        print(f'Skipping {modelname} (baseline has no kernels)')
        return
    # Load data
    ds = IO.get_split('train',CFG.filedir)
    fielddata,localdata,targetdata = Data.totensor(ds,CFG.fieldvars,CFG.localvars,CFG.targetvar)
    quadweights_full = torch.from_numpy(ds['quadweights'].values.astype(np.float32))
    # Build and load model
    nfieldvars,nlocalvars = len(CFG.fieldvars),len(CFG.localvars)
    nlevs = fielddata.shape[3]
    patchcfg = modelcfg['patch']
    patch = Patch(patchcfg['radius'],patchcfg['maxlevs'],patchcfg['timelag'])
    patchshape = patch.shape(nlevs)
    model = build(modelcfg,patchshape,nfieldvars,nlocalvars)
    IO.get_model(model,modelname,CFG.modeldir)
    # Extract and save weights
    quadweights = quadweights_full[:patchshape[0],:patchshape[1],:patchshape[2],:patchshape[3]]
    weights = model.kernellayer.weights(quadweights).detach().cpu().numpy()
    os.makedirs(savedir,exist_ok=True)
    np.save(os.path.join(savedir,f'{modelname}_weights.npy'),weights)
    print(f'Saved {modelname} weights: {weights.shape}')
    # Extract and save features
    if extract_features:
        centers = patch.centers(targetdata,ds.lat.values,ds.lon.values,CFG.latrange,CFG.lonrange)
        dataset = SampleDataset(fielddata,localdata,targetdata,centers,patch,modelcfg.get('uselocal',True))
        loader = DataLoader(dataset,batch_size=CFG.batchsize,shuffle=False,num_workers=4)
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        quadweights = quadweights.to(device)
        features_list = []
        targets_list = []
        print(f'Extracting features from {len(centers)} samples...')
        with torch.no_grad():
            for batch in loader:
                patchbatch = batch['patch'].to(device)
                features = model.kernellayer(patchbatch,quadweights)
                features_list.append(features.cpu().numpy())
                targets_list.append(batch['target'].cpu().numpy())
        features = np.concatenate(features_list,axis=0)
        targets = np.concatenate(targets_list,axis=0)
        np.save(os.path.join(savedir,f'{modelname}_features.npy'),features)
        np.save(os.path.join(savedir,f'{modelname}_targets.npy'),targets)
        print(f'Saved {modelname} features: {features.shape}')
    # Save metadata
    metadata = {'fieldvars':CFG.fieldvars,'patchshape':patchshape,'type':modelcfg['type'],
                'nkernels':modelcfg.get('nkernels',0)}
    np.save(os.path.join(savedir,f'{modelname}_metadata.npy'),metadata,allow_pickle=True)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract kernel weights and features from trained models.')
    parser.add_argument('--models',type=str,default='all',help='Comma-separated model names or "all"')
    parser.add_argument('--savedir',type=str,default='../../data/weights',help='Output directory')
    parser.add_argument('--no-features',action='store_true',help='Skip feature extraction')
    args = parser.parse_args()
    CFG = Config()
    requested = [m.strip() for m in args.models.split(',')] if args.models!='all' else None
    for modelcfg in CFG.models:
        modelname = modelcfg['name']
        if requested is not None and modelname not in requested:
            continue
        extract(modelname,args.savedir,extract_features=not args.no_features)
    print('Extraction complete.')