#!/usr/bin/env python

import json


class Config:

    def __init__(self, path='configs.json'):
        '''
        Purpose: Load configs.json and expose commonly used blocks/paths as attributes.
        '''
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.paths     = config['paths']
            self.variables = config['variables']
            self.domain    = config['domain']
            self.training  = config['training']
            self.models    = config['models']

    @property
    def splitdir(self):
        return self.paths['splitdir']

    @property
    def modeldir(self):
        return self.paths['modeldir']

    @property
    def featuredir(self):
        return self.paths['featuredir']

    @property
    def predictionsdir(self):
        return self.paths['predictionsdir']

    @property
    def fieldvars(self):
        return self.variables['fieldvars']

    @property
    def localvars(self):
        return self.variables['localvars']

    @property
    def targetvar(self):
        return self.variables['targetvar']

    @property
    def latrange(self):
        latmin,latmax = self.domain['latrange']
        return float(latmin),float(latmax)

    @property
    def lonrange(self):
        lonmin,lonmax = self.domain['lonrange']
        return float(lonmin),float(lonmax)

    @property
    def seed(self):
        return int(self.training['seed'])

    @property
    def workers(self):
        return int(self.training['workers'])

    @property
    def epochs(self):
        return int(self.training['epochs'])

    @property
    def batchsize(self):
        return int(self.training['batchsize'])

    @property
    def learningrate(self):
        return float(self.training['learningrate'])

    @property
    def patience(self):
        return int(self.training['patience'])

    @property
    def criterion(self):
        return self.training['criterion']