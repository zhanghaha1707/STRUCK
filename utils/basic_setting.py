'''
Author: Zhang haha
Date: 2022-07-29 02:14:39
LastEditTime: 2023-01-22
Description: basic settings

'''
import logging
import sys

import numpy  as np
import torch
import os

import shutil
import tarfile
import re
import pycparser
from tqdm import tqdm
import pickle
import random
import yaml
import time

def set_logger(logger,level=logging.INFO,log_path="../save/logs/record.log"):
    logger.setLevel(level)
    ori_path=log_path
    log_path=log_path.split('/')
    if len(log_path)>1:
        file_dir=('/').join(log_path[:-1])
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
    if not os.path.isfile(ori_path):
        with open(ori_path,"w"):
            pass

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S')  
    fh = logging.FileHandler(ori_path, encoding='utf-8',mode='a')  
    fh.setLevel(logging.DEBUG)  
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
  
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def set_attack_logger(logger,level=logging.INFO,log_path="../save/logs/record.log",write_mode='a'):
   
    logger.setLevel(level)
    ori_path=log_path
    log_path=log_path.split('/')
    if len(log_path)>1:
        file_dir=('/').join(log_path[:-1])
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
    if not os.path.isfile(ori_path):
        with open(ori_path,"w"):
            pass
   
    fh = logging.FileHandler(ori_path, encoding='utf-8',mode=write_mode)  
    fh.setLevel(logging.DEBUG)  
    
    logger.addHandler(fh)
 
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    
    logger.addHandler(ch)
def set_seed(seed=2022,n_gpu=0):
   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:# 根据使用GPU的数目
        torch.cuda.manual_seed_all(seed)

def bestGPU(gpu_verbose=False, **w):
    import GPUtil
    import numpy as np

    Gpus = GPUtil.getGPUs()
    Ngpu = 4
    mems, loads = [], []
    for ig, gpu in enumerate(Gpus):
        memUtil = gpu.memoryUtil * 100
        load = gpu.load * 100
        mems.append(memUtil)
        loads.append(load)
        if gpu_verbose: print(f'gpu-{ig}:   Memory: {memUtil:.2f}%   |   load: {load:.2f}% ')
    bestMem = np.argmin(mems)
    bestLoad = np.argmin(loads)
    best = bestMem
    if gpu_verbose: print(f'//////   Will Use GPU - {best}  //////')

    return int(best)
def set_device(config):
    if config['device'] in [0,1]:
        n_gpu=1
       
        device = torch.device(f"cuda:{bestGPU(True)}" if torch.cuda.is_available() else "cpu")
    elif config['device']==-1:
        device=torch.device('cpu')
        n_gpu=0
    else:
        n_gpu = torch.cuda.device_count()
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return n_gpu,device
class myConfig(object):
    '''设置类'''
    def __init__(self,path='../config/trainning_config.yaml') -> None:
        self.path=path
        self.config=self.get_config(path)
    @staticmethod    
    def get_config(config_path='../config/trainning_config.yaml',target=None):    
      
        with open(config_path, "r") as setting:
            config = yaml.load(setting, Loader=yaml.FullLoader)
        if target!=None:
            return {key:value for key, value in config.items() if key in target}
        return config
    @classmethod
    def show_config(cls,config,name="top",logger=None):
        
        if logger is not None:
            cur_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
            logger.info(f"now time is: {cur_time}")
            logger.info(f"**************** {name.upper()} CONFIGURATION ****************")
        else: print(f"**************** {name.upper()} CONFIGURATION ****************")
        for key in sorted(config.keys()):
            val = config[key]
            if not isinstance(val,dict):
                keystr = "{}".format(key) + (" " * (24 - len(key)))
                if logger is not None:
                    logger.info("{} -->   {}".format(keystr, val))
                else: print("{} -->   {}".format(keystr, val))
            else:
                cls.show_config(val,name=key,logger=logger)
        if logger is not None:
            logger.info(f"**************** {name.upper()} CONFIGURATION ****************")
    
    def update_config(self,new_config):
       
        self.config.update(new_config)
        with open(self.path, "w") as setting:
            yaml.dump(self.config,setting)
        return self.config
class single_code_node_attack_result(object):
   
    def __init__(self,succ:bool=False,result_type:int=2,ori_code=None,adv_code=None,new_pred=None,node_changes:dict=None,names_to_important_score:dict=None,nb_changed_var=None,nb_changed_pos=None) -> None:
       
        self.succ=succ
        self.result_type=result_type
        self.ori_code=ori_code
        self.node_len=len(ori_code)
        self.adv_code=adv_code
        self.new_pred=new_pred
        self.node_changes=node_changes
        self.names_to_important_score=names_to_important_score
        self.nb_changed_var=nb_changed_var
        self.nb_changed_pos=nb_changed_pos
    def update(self,succ=True,result_type=1,adv_code=None,new_pred=None,names_to_important_score=None,node_changes=None,nb_changed_var=0,nb_changed_pos=0):
        self.succ=succ
        self.result_type=result_type
        self.adv_code=adv_code
        self.new_pred=new_pred
        self.node_changes=node_changes
        self.names_to_important_score=names_to_important_score
        self.nb_changed_var=nb_changed_var
        self.nb_changed_pos=nb_changed_pos
    def show_result(self,logger=None):
        if self.result_type==0:
            resutl_str="SUCC! Original mistake."
        elif self.result_type==1:
            resutl_str=f"SUCC!\t number of changed variables is {self.nb_changed_var}, The corresponding modifications are as follows:\n\t\t\t"
            for var in self.node_changes.keys():
                if var!=self.node_changes[var]:
                    resutl_str+=f"{var}=>{self.node_changes[var]}\t"
            resutl_str+=f'\n\tthe perturbation of node attack is {self.nb_changed_pos/self.node_len} '
        if logger is not None:
            logger.info(resutl_str)
        else:
            print(resutl_str)

if __name__=="__main__":
    mycfg=myConfig()
    config=mycfg.config
    mycfg.show_config(config)

