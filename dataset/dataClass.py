'''
Author: Zhang haha
Date: 2022-08-01 02:28:14
LastEditTime: 2023-01-22
Description: prepare data for model input
'''

from typing import Counter
import torch
from torch.utils.data import Dataset
import gzip
import pickle
import numpy as np
import os
from CodeFeature import CodeFeatures,CodeFeatures_tbcnn,CodeFeatures_astnn,CodeFeatures_codebert,CodeFeatures_graphcodebert,CodeFeatures_ggnn
from GraphDataset import GGNNDataset_s
from vocabClass import VocabModel, VocabModel_codebert, VocabModel_ggnn_without_seq, VocabModel_tbcnn,VocabModel_astnn,VocabModel_graphcodebert
import sys
sys.path.append("..")
from utils.tools import read_pkl,save_pkl
from utils.basic_setting import *
from random import shuffle,sample
from tqdm.contrib import tzip 

class LSTMDataset(Dataset):
    def __init__(self,vocab,type='train',model_name='lstm',restore=False,config=None,is_enhance=False,enhance_type=None,sample_nums=None):
        '''
        Description: Initializes the data set
        '''
        self.vocab=vocab
        self.config=config
        self.type=type
        self.model_name=model_name
        self.restore=restore
        self.max_len=self.config[model_name]['max_len']
        if not is_enhance :
            self.data=self.load_data()
        else:
            
            if sample_nums is not None and sample_nums>0:
                enhance_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_{str(sample_nums)}.pkl")
                adv_datas_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_retrain_{str(sample_nums)}_{config[model_name]['enhance_size']}.pkl")
            else:
                enhance_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}.pkl")
                adv_datas_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_retrain_{config[model_name]['enhance_size']}.pkl")
            self.data=self.load_adv_date(adv_datas_path,enhance_path,self.config[model_name]["enhance_size"])
    def load_data(self):
        
        if os.path.isfile(self.config[self.model_name][self.type]) and not self.restore:
            return read_pkl(self.config[self.model_name][self.type])
        else:
            with gzip.open(self.config['raw_data_path'][self.type]) as f:
                raw_data=pickle.load(f)
            ids=raw_data['id']
            codes=raw_data['code']
            labels=raw_data['label']
            # Encapsulate data with codeFeature classes
            features = [CodeFeatures(code_id,code,label,self.vocab,self.max_len) for code_id,code,label in tzip(ids,codes,labels)]
            save_pkl(features,self.config[self.model_name][self.type])
            return features
    def load_adv_date(self,adv_datas_path,enhance_path,enhance_size):
        if os.path.isfile(adv_datas_path) and not self.restore:
            return read_pkl(adv_datas_path)
        else:
            # Create a data set
            # Load the raw data
            adv_datas=self.load_data()
            enhance_date=read_pkl(enhance_path)
            if enhance_size == 'None':
                enhance_size=len(enhance_date)
            else:
                enhance_size=min(enhance_size,len(enhance_date))
            add_datas=sample(enhance_date,enhance_size)
            adv_datas+=add_datas
            adv_datas=sample(adv_datas,len(adv_datas))
            save_pkl(adv_datas,adv_datas_path)
            return adv_datas
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item): 
        return (torch.tensor(self.data[item].input_idx),      
                torch.tensor(self.data[item].mask),
                torch.tensor(self.data[item].label),)

class TBCNNDataset(object):
    def __init__(self,vocab,type='train',model_name='tbcnn',restore=False,config=None,is_enhance=False,enhance_type=None,sample_nums=None):
        
        self.vocab=vocab
        self.config=config
        self.type=type 
        self.model_name=model_name
        self.restore=restore
        if not is_enhance :
            self.data=self.load_data()
        else:
            
            if sample_nums is not None and sample_nums>0:
                enhance_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_{str(sample_nums)}.pkl")
                adv_datas_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_retrain_{str(sample_nums)}_{config[model_name]['enhance_size']}.pkl")
            else:
                enhance_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}.pkl")
                adv_datas_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_retrain_{config[model_name]['enhance_size']}.pkl")
            self.data=self.load_adv_date(adv_datas_path,enhance_path,self.config[model_name]["enhance_size"])
    
    def load_data(self):
        if os.path.isfile(self.config[self.model_name][self.type]) and not self.restore:
            return read_pkl(self.config[self.model_name][self.type])
        else:
            
            with gzip.open(self.config['raw_data_path'][self.type]) as f:
                raw_data=pickle.load(f)
            ids=raw_data['id']
            codes=raw_data['code']
            labels=raw_data['label']
            
            features = [CodeFeatures_tbcnn(code_id,code,label,self.vocab) for code_id,code,label in tzip(ids,codes,labels)]
            save_pkl(features,self.config[self.model_name][self.type])
            return features
    def load_adv_date(self,adv_datas_path,enhance_path,enhance_size):
        if os.path.isfile(adv_datas_path) and not self.restore:
            return read_pkl(adv_datas_path)
        else:
            
            adv_datas=self.load_data()
            enhance_date=read_pkl(enhance_path)
            if enhance_size == 'None':
                enhance_size=len(enhance_date)
            else:
                enhance_size=min(enhance_size,len(enhance_date))
            add_datas=sample(enhance_date,enhance_size)
            adv_datas+=add_datas
            adv_datas=sample(adv_datas,len(adv_datas))
            save_pkl(adv_datas,adv_datas_path)
            return adv_datas
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item): 
        return self.data[item].graph

class ASTNNDataset(object):
    def __init__(self,vocab:VocabModel_astnn,type='train',model_name='astnn',restore=False,config=None,is_enhance=False,enhance_type=None,sample_nums=None):
        
        self.vocab=vocab
        self.config=config
        self.type=type 
        self.model_name=model_name
        self.restore=restore
        if not is_enhance :
            self.data=self.load_data()
        else:
            
            if sample_nums is not None and sample_nums>0:
                enhance_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_{str(sample_nums)}.pkl")
                adv_datas_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_retrain_{str(sample_nums)}_{config[model_name]['enhance_size']}.pkl")
            else:
                enhance_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}.pkl")
                adv_datas_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_retrain_{config[model_name]['enhance_size']}.pkl")
            self.data=self.load_adv_date(adv_datas_path,enhance_path,self.config[model_name]["enhance_size"])
        self.in_idx=list(range(self.__len__()))
    
    def load_data(self):
        if os.path.isfile(self.config[self.model_name][self.type]) and not self.restore:
            return read_pkl(self.config[self.model_name][self.type])
        else:
            
            with gzip.open(self.config['raw_data_path'][self.type]) as f:
                raw_data=pickle.load(f)
            ids=raw_data['id']
            codes=raw_data['code']
            labels=raw_data['label']
           
            features = [CodeFeatures_astnn(code_id,code,label,self.vocab) for code_id,code,label in tzip(ids,codes,labels)]
            save_pkl(features,self.config[self.model_name][self.type])
            return features
    def load_adv_date(self,adv_datas_path,enhance_path,enhance_size):
        if os.path.isfile(adv_datas_path) and not self.restore:
            return read_pkl(adv_datas_path)
        else:
           
            adv_datas=self.load_data()
            enhance_date=read_pkl(enhance_path)
            if enhance_size == 'None':
                enhance_size=len(enhance_date)
            else:
                enhance_size=min(enhance_size,len(enhance_date))
            add_datas=sample(enhance_date,enhance_size)
            adv_datas+=add_datas
            adv_datas=sample(adv_datas,len(adv_datas))# 打乱顺序
            save_pkl(adv_datas,adv_datas_path)
            return adv_datas
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item): 
        
        return (self.data[item].tree_in,
                self.data[item].label)
    def get_batch(self,step,batch_size,shuffle=False):
        if shuffle:
            self.in_idx=sample(range(self.__len__()), self.__len__())
        trees_in,labels=[],[]
        start_idx=step*batch_size
        for idx in self.in_idx[start_idx:start_idx+batch_size]:
            (tree_in,label) = self.__getitem__(idx)
            trees_in.append(tree_in)
            labels.append(label)
        return (trees_in,torch.tensor(labels))
    @staticmethod
    def get_model_input(codefeatures_list):
        if not isinstance(codefeatures_list,list):
            codefeatures_list=[codefeatures_list]
        trees_in,labels=[],[]
        for codefeatures in codefeatures_list:
            trees_in.append(codefeatures.tree_in)
            labels.append(codefeatures.label)
        return (trees_in,torch.tensor(labels))
class CodeBertDataset(object):
    def __init__(self,vocab,type='train',model_name='codebert',restore=False,config=None,is_enhance=False,enhance_type=None,sample_nums=None):
    
        self.vocab=vocab
        self.type=type 
        self.model_name=model_name
        self.restore=restore
        self.config=config
        if not is_enhance :
            self.data=self.load_data()
        else:
            
            if sample_nums is not None and sample_nums>0:
                enhance_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_{str(sample_nums)}.pkl")
                adv_datas_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_retrain_{str(sample_nums)}_{config[model_name]['enhance_size']}.pkl")
            else:
                enhance_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}.pkl")
                adv_datas_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_retrain_{config[model_name]['enhance_size']}.pkl")
            self.data=self.load_adv_date(adv_datas_path,enhance_path,self.config[model_name]["enhance_size"])
    def load_data(self):
        if os.path.isfile(self.config[self.model_name][self.type]) and not self.restore:
            return read_pkl(self.config[self.model_name][self.type])
        else:
            
            with gzip.open(self.config['raw_data_path'][self.type]) as f:
                raw_data=pickle.load(f)
            ids=raw_data['id']
            codes=raw_data['code']
            labels=raw_data['label']
            
            features = [CodeFeatures_codebert(code_id,code,label,self.vocab) for code_id,code,label in tzip(ids,codes,labels)]
            save_pkl(features,self.config[self.model_name][self.type])
            return features
    def load_adv_date(self,adv_datas_path,enhance_path,enhance_size):
        if os.path.isfile(adv_datas_path) and not self.restore:
            return read_pkl(adv_datas_path)
        else:
            
            adv_datas=self.load_data()
            enhance_date=read_pkl(enhance_path)
            if enhance_size == 'None':
                enhance_size=len(enhance_date)
            else:
                enhance_size=min(enhance_size,len(enhance_date))
            add_datas=sample(enhance_date,enhance_size)
            adv_datas+=add_datas
            adv_datas=sample(adv_datas,len(adv_datas))
            save_pkl(adv_datas,adv_datas_path)
            return adv_datas
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item): 
        
        return (torch.tensor(self.data[item].input_ids),
                torch.tensor(self.data[item].label))
    @staticmethod
    def get_model_input(codefeatures_list):
        if not isinstance(codefeatures_list,list):
            codefeatures_list=[codefeatures_list]
        input_ids,labels=[],[]
        for codefeatures in codefeatures_list:
            input_ids.append(codefeatures.input_ids)
            labels.append(codefeatures.label)
        return (torch.tensor(input_ids),
                torch.tensor(labels))
    

class GraphCodeBertDataset(Dataset):
    def __init__(self, vocab,type='train',model_name='graphcodebert',restore=False,config=None,is_enhance=False,enhance_type=None,sample_nums=None):
        
        self.vocab=vocab
        self.type=type 
        self.model_name=model_name
        self.restore=restore
        self.config=config
        if not is_enhance :
            self.data=self.load_data()
        else:
            
            if sample_nums is not None and sample_nums>0:
                enhance_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_{str(sample_nums)}.pkl")
                adv_datas_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_retrain_{str(sample_nums)}_{config[model_name]['enhance_size']}.pkl")
            else:
                enhance_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}.pkl")
                adv_datas_path=os.path.join(self.config[model_name]["enhance_path"],f"{type}_{enhance_type}_retrain_{config[model_name]['enhance_size']}.pkl")
            self.data=self.load_adv_date(adv_datas_path,enhance_path,self.config[model_name]["enhance_size"])
    def load_data(self):
        if os.path.isfile(self.config[self.model_name][self.type]) and not self.restore:
            return read_pkl(self.config[self.model_name][self.type])
        else:
            
            with gzip.open(self.config['raw_data_path'][self.type]) as f:
                raw_data=pickle.load(f)
            ids=raw_data['id']
            codes=raw_data['code']
            labels=raw_data['label']
            
            features = [CodeFeatures_graphcodebert(code_id,code,label,self.vocab,config=self.config) for code_id,code,label in tzip(ids,codes,labels)]
            save_pkl(features,self.config[self.model_name][self.type])
            return features
    def load_adv_date(self,adv_datas_path,enhance_path,enhance_size):
        if os.path.isfile(adv_datas_path) and not self.restore:
            return read_pkl(adv_datas_path)
        else:
            
            adv_datas=self.load_data()
            enhance_date=read_pkl(enhance_path)
            if enhance_size == 'None':
                enhance_size=len(enhance_date)
            else:
                enhance_size=min(enhance_size,len(enhance_date))
            add_datas=sample(enhance_date,enhance_size)
            adv_datas+=add_datas
            adv_datas=sample(adv_datas,len(adv_datas))
            save_pkl(adv_datas,adv_datas_path)
            return adv_datas
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function  
        attn_mask= np.zeros((self.config['graphcodebert']['code_length']+self.config['graphcodebert']['data_flow_length'],
                        self.config['graphcodebert']['code_length']+self.config['graphcodebert']['data_flow_length']),dtype=bool) 
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.data[item].position_idx])
        max_length=sum([i!=1 for i in self.data[item].position_idx])
        #sequence can attend to sequence 
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.data[item].input_ids):
            if i in [0,2]:# 0-cls,2->sep
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.data[item].dfg_to_code):
            if a<node_index and b<node_index:
                
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes，
        for idx,nodes in enumerate(self.data[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.data[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
                    
                      
                    
        return (torch.tensor(self.data[item].input_ids),
                torch.tensor(self.data[item].position_idx),
                torch.tensor(attn_mask),                 
                torch.tensor(self.data[item].label))
    @staticmethod
    def generate_atten_mask(codefeatures,code_length=256,flow_length=64):
        #calculate graph-guided masked function 
        attn_mask= np.zeros((code_length+flow_length,code_length+flow_length),dtype=bool) 
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in codefeatures.position_idx])
        max_length=sum([i!=1 for i in codefeatures.position_idx])
        #sequence can attend to sequence 
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens 
        for idx,i in enumerate(codefeatures.input_ids):
            if i in [0,2]:# 0-cls,2->sep
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(codefeatures.dfg_to_code):
            if a<node_index and b<node_index:# 
                
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes，
        for idx,nodes in enumerate(codefeatures.dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(codefeatures.position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
        return attn_mask 
    @staticmethod
    def generate_atten_mask_from_input(input_ids,position_idx,dfg_to_code,dfg_to_dfg,code_length=256,flow_length=64):
        #calculate graph-guided masked function  
        attn_mask= np.zeros((code_length+flow_length,code_length+flow_length),dtype=bool) 
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in position_idx])
        max_length=sum([i!=1 for i in position_idx])
        #sequence can attend to sequence 
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens 
        for idx,i in enumerate(input_ids):
            if i in [0,2]:# 0-cls,2->sep
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(dfg_to_code):
            if a<node_index and b<node_index:# 
                
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes，
        for idx,nodes in enumerate(dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
        return attn_mask 
    @classmethod
    def get_model_input(cls,codefeatures_list,device):
        if not isinstance(codefeatures_list,list):
            codefeatures_list=[codefeatures_list]
        input_ids,position_idxs,attn_mask,labels=[],[],[],[]
        for codefeatures in codefeatures_list:
            input_ids.append(codefeatures.input_ids)
            position_idxs.append(codefeatures.position_idx)
            attn_mask.append(cls.generate_atten_mask(codefeatures))
            labels.append(codefeatures.label)
        return (torch.tensor(input_ids).to(device),
                torch.tensor(position_idxs).to(device),
                torch.tensor(np.array(attn_mask)).to(device),
                torch.tensor(labels).to(device))
        
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                        default='lstm', help="dataset name, choose from [lstm,gru,codebert,graphcodebert]")
    parser.add_argument('--is_restore', type=bool,
                        default=False, help="Whether to regenerate the dataclass")
    args = parser.parse_args()
    restore=args.is_restore
    model_name=args.name
    vocab_map = {
            "lstm": VocabModel, 
            'gru': VocabModel, 
            'codebert': VocabModel_codebert, 
            "graphcodebert": VocabModel_graphcodebert, 
            'ggnn_simple':VocabModel_ggnn_without_seq}
    dataset_map = {
                    "lstm": LSTMDataset, 
                    "gru": LSTMDataset, 
                    "tbcnn": TBCNNDataset, 
                    'astnn': ASTNNDataset,
                    'codebert': CodeBertDataset, 
                    'graphcodebert': GraphCodeBertDataset, 
                    'ggnn':GGNNDataset_s}
    vocab=vocab_map[model_name]()
    config=myConfig().config
    
    train=dataset_map[model_name](vocab,'train',restore=restore,config=config)
    dev=dataset_map[model_name](vocab,'dev',restore=restore,config=config)
    test=dataset_map[model_name](vocab,'test',restore=restore,config=config)
    
