#!/usr/bin/env python 

import json 

class Config: 
    '''
    Purpose: Load configs.json and expose commonly used blocks/paths as attributes.
    ''' 
    def __init__(self,path='configs.json'): 
        with open(path,'r',encoding='utf-8') as f: 
            config = json.load(f) 
            self.paths  = config['paths'] 
            self.attrs  = config['attrs']
            self.domain = config['domain']
            self.splits = config['splits']
            
    @property 
    def rawdir(self):
        return self.paths['rawdir']
        
    @property 
    def interimdir(self):
        return self.paths['interimdir']
        
    @property 
    def splitdir(self):
        return self.paths['splitdir']
        
    @property 
    def author(self):
        return self.attrs['author'] 
        
    @property 
    def email(self):
        return self.attrs['email'] 
                    
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
        trainmin,trainmax = self.splits['trainrange'] 
        return int(trainmin),int(trainmax) 

    @property 
    def validrange(self): 
        validmin,validmax = self.splits['validrange'] 
        return int(validmin),int(validmax) 

    @property 
    def testrange(self): 
        testmin,testmax = self.splits['testrange'] 
        return int(testmin),int(testmax) 