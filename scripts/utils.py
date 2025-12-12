#!/usr/bin/env python 

import json 

class Config: 

    def __init__(self,path='configs.json'): 
        '''
        Purpose: Load configurations from a JSON file and expose commonly used blocks/paths as attributes.
        '''
        with open(path,'r',encoding='utf-8') as f: 
            config = json.load(f) 
            self.filepaths = config['filepaths'] 
            self.metadata  = config['metadata']
            self.domain    = config['domain']
            self.splits    = config['splits']
            self.variables = config['splits']
            self.training  = config['training']
            
    @property 
    def rawdir(self):
        return self.paths['ra']
        
    @property 
    def interimdir(self):
        return self.paths['interim']
        
    @property 
    def splitsdir(self):
        return self.paths['splits']
        
    @property 
    def predsdir(self):
        return self.paths['predictions']
    
    @property 
    def featsdir(self):
        return self.paths['features']

    @property 
    def weightsdir(self):
        return self.paths['weights']

    @property 
    def modelsdir(self):
        return self.paths['models']

    @property 
    def author(self):
        return self.metadata['author'] 
        
    @property 
    def email(self):
        return self.metadata['email'] 
                    
    @property 
    def latrange(self): 
        latmin,latmax = self.domain['latrange'] 
        return float(latmin),float(latmax) 
        
    @property 
    def lonrange(self): 
        lonmin,lonmax = self.domain['lonrange'] 
        return float(lonmin),float(lonmax) 
            
    @property 
    def levrange(self): 
        levmin,levmax = self.domain['levrange'] 
        return float(levmin),float(levmax) 
            
    @property 
    def years(self): 
        return self.domain['years']
            
    @property 
    def months(self): 
        return self.domain['months']

    @property 
    def trainrange(self): 
        trainstart,trainend = self.splits['trainyears'] 
        return int(trainstart),int(trainend) 

    @property 
    def validrange(self): 
        validstart,validend = self.splits['validyears'] 
        return int(validstart),int(validend) 

    @property 
    def testrange(self): 
        teststart,testend = self.splits['testyears'] 
        return int(teststart),int(testend)

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
    def projectname(self):
        return self.training['projectname']
        
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