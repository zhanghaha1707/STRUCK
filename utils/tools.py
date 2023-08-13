'''
Author: Zhang haha
Date: 2022-07-29 02:35:05
LastEditTime: 2023-01-21
Description: Some tool functions
'''
import gzip
import pickle
import os
import sys
import torch
import numpy as np
import random
sys.path.append("../dataset/")

def check_folder(folder):

    if not os.path.exists(folder):
            os.mkdir(folder)

def save_data(root,data,name,new_floder=False,logger=None):
    if new_floder:
        folder=os.path.join(root,name)
        check_folder(folder)
        file_path=os.path.join(folder,name+'_raw.pkl')
        with gzip.open(file_path,'wb') as f:
            pickle.dump(data,f)
    else:
        file_path=os.path.join(root,name+'.pkl')
        with gzip.open(file_path,'wb') as f:
            pickle.dump(data,f)
    if logger is not None:
        logger.Info(f'')
def read_oj_pkl(path='../data/oj.pkl.gz'):
    with gzip.open(path, "rb") as f:
        oj = pickle.load(f)
    return oj
def read_pkl(path):
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    return data
def save_pkl(data,path):
    ori_path=path
    path=path.split('/')
    if len(path)>1:
        file_dir=('/').join(path[:-1])
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
    if not os.path.isfile(ori_path):
        with open(ori_path,"w"):
            pass
    with gzip.open(ori_path, "wb") as f:
        pickle.dump(data,f)
def const_norm(token):
    if token[0] == '\'' and token[-1] == '\'':
        return "<char>"
    elif token[0] == '"' and token[-1] == '"':
        return "<str>"
    try:
        int(token)
        return "<int>"
    except ValueError:
        try:
            float(token)
            return "<fp>"
        except ValueError:
            return token
def get_input_from_codefeatures_list(codefeatures_list,device):
    inputs_ids=[]
    masks=[]
    labels=[]
    
    for codefeatures in codefeatures_list:
        (inputs_id,mask,label)=get_input_from_codefeatures(codefeatures,device)
        inputs_ids.append(inputs_id)
        masks.append(mask)
        labels.append(label)

    return torch.stack(inputs_ids),torch.stack(masks),torch.stack(labels)
def get_input_from_codefeatures(codefeatures,device):
    return (torch.tensor(codefeatures.input_idx).to(device),      
            torch.tensor(codefeatures.mask).to(device),
            torch.tensor(int(codefeatures.label)).to(device),) 

def create_mask(x, N):
    x = x.data# 
    mask = np.zeros((x.size(0), N))
    for i in range(x.size(0)):
        mask[i, :x[i]] = 1
    return torch.Tensor(mask)
def weight_choice(key_weight):
    if isinstance(key_weight,dict):
        key_weight=[(key,item) for key,item in key_weight.items()]
    sum_weight=sum([weight for _,weight in key_weight])
    t = random.uniform(0, sum_weight)
    for key, weight in key_weight:
        t -= weight
        if t < 0:
            return key  
