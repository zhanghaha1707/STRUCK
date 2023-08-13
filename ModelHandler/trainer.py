'''
Author: Zhang
Date: 2022-08-14
LastEditTime: 2023-01-23
Description: 整合所有模型的训练函数
             不同点：模型的输入设置

'''


import torch
import logging
from tqdm import tqdm
import sys
sys.path.append("../dataset/")


sys.path.append("..")
from model.LSTM import LSTMClassifier
from model.GRU import GRUClassifier
from utils.basic_setting import set_device, set_logger,myConfig,set_seed
from torch import optim
from torch.utils.data import Dataset,DataLoader, RandomSampler,SequentialSampler
from transformers import get_linear_schedule_with_warmup
import argparse
import numpy as np
import os
from config.support import *
# from torch_geometric.data import DataLoader as pygDataLoader
from sklearn.metrics import accuracy_score
from dataset.GraphDataset import Batch_s
class ModelHandler(object):
   
    def __init__(self,args) -> None:
        self.args=args 
        self.basic_init()
    
    def basic_init(self):
       
        set_seed()
        mycfg=myConfig() 
        self.config=mycfg.config
        self.model_name=self.args.train_model
        self.model_config=self.config[self.model_name]
        if args.train_model=='ggnn' and args.graph_type is not None and self.model_config['graph_type']!=args.graph_type:
            self.model_config['graph_type']=args.graph_type
        if self.model_name=='ggnn' and not self.model_config['graph_type'].startswith('ggnn'):
            self.model_config['model_path']=self.model_config['model_path'].replace('ggnn',self.model_config['graph_type'])
        
        self.logger=logging.getLogger(__name__)
        if self.args.is_enhance:
           self.model_config['log_path']= self.model_config['log_path'].replace('train_log.log',f'train_{self.args.enhance_type}_enhance{self.args.enhance_size}.log')
        set_logger(self.logger,log_path=self.model_config['log_path'])
        # vocab
        self.vocab=vocab_map[self.model_name]() 
       
        n_gpu,device=set_device(self.config)
        self.device=device
        self.model_config['device']=device
        self.model_config['n_gpu']=n_gpu
        self.model_config['vocab']=self.vocab
        self.model_config['enhance_size']=self.args.enhance_size
        self.config[self.model_name]['enhance_size']=self.args.enhance_size
        if self.model_name =='astnn':
            if self.model_config['using_word2vec_embedding']:
                self.model_config['word2vec_weight']=self.vocab.embeddings
            else: 
                self.model_config['word2vec_weight']=None
            self.model_config['vocab_size']=self.vocab.vocab_size
            self.model=support_model[self.model_name](**self.model_config)
        else:
            self.model=support_model[self.model_name](**self.model_config)
        
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
       
        self.model.to(device)
        
    def run(self):
        if self.args.train:
            self.run_train()
        elif self.args.val:
            self.run_test('dev')
        if self.args.test:
            self.run_test('test')
    def run_train(self):        
        
        self.logger.debug(f"start training,the logs about traiing will be saved in {self.model_config['log_path']} ")
        self.logger.warning(f"++++++++++++++++++++New+++++++++++++++++++++++++")
        
        myConfig.show_config(config=self.model_config,name=self.model_name,logger=self.logger)
        
        output_dir = self.model_config['model_path']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if args.is_enhance is True:
            if args.sample_nums >0:
                file_name=f"best_{self.args.enhance_type}_sample{str(args.sample_nums)}_{self.args.enhance_size}.pt"
            else:
                file_name=f'best_{self.args.enhance_type}_{self.args.enhance_size}.pt'
                
            parameter_path=os.path.join(output_dir,file_name)
        else:
            parameter_path=os.path.join(output_dir,'best.pt')
        self.model_config['parameter_path']=parameter_path
        if self.args.pretrain:
            model_parameter=torch.load(self.model_config['parameter_path'])
            if self.model_config['n_gpu'] in [1,2]:# GPU
                if list(model_parameter.keys())[0].startswith('module'):
                    if not isinstance(self.model,torch.nn.DataParallel):
                        self.model = torch.nn.DataParallel(self.model,device_ids=[0, 1])
                    self.model.load_state_dict(model_parameter)
                else:
                    self.model.load_state_dict(model_parameter)
            else: # cpu
                self.model.load_state_dict(torch.load(
                    model_parameter, map_location='cpu'))  # cpu上
      
        train_dataset=support_dataset[self.model_name](self.vocab,'train',model_name=self.model_name,config=self.config,is_enhance=self.args.is_enhance,enhance_type=self.args.enhance_type,sample_nums=self.args.sample_nums)# 数据加载需要全局的config
        
        dev_dataset=support_dataset[self.model_name](self.vocab,'dev',config=self.config)
        # dev_dataset=support_dataset[self.model_name](self.vocab,'test',config=self.config)
        if self.model_name in  ['ggnn','ggnn_simple']:
            self.train_gnn(self.model,train_dataset,dev_dataset,self.logger,**self.model_config)
        elif self.model_name in ['lstm','gru']:
            self.train_lstm(self.model,train_dataset,dev_dataset,self.logger,**self.model_config)
        elif self.model_name == 'codebert':
            self.train_codebert(self.model,train_dataset,dev_dataset,self.logger,**self.model_config)
        elif self.model_name == 'graphcodebert':
            self.train_graphcodebert(self.model,train_dataset,dev_dataset,self.logger,**self.model_config)
        elif self.model_name == 'astnn':
            self.train_astnn(self.model,train_dataset,dev_dataset,self.logger,**self.model_config)
        elif self.model_name == 'tbcnn':
            self.train_tbcnn(self.model,train_dataset,dev_dataset,self.logger,**self.model_config)
    def run_test(self,type='dev'):
        '''
        Description: for dev or test
        param mode [str]:dev or test
        '''
        try:
            parameter_path=self.model_config['parameter_path']
        except:
            output_dir = self.model_config['model_path']
            if args.is_enhance is True:
                if args.sample_nums >0:
                    file_name=f"best_{self.args.enhance_type}_sample{str(args.sample_nums)}_{self.args.enhance_size}.pt"
                else:
                    file_name=f'best_{self.args.enhance_type}_{self.args.enhance_size}.pt'
                    
                parameter_path=os.path.join(output_dir,file_name)
            else:
                parameter_path=os.path.join(output_dir,'best.pt')
      
        self.logger.info(f'load parameter from {parameter_path} for testing')
        model_parameter=torch.load(parameter_path)
        if self.model_config['n_gpu'] in [1,2]:# GPU
            if list(model_parameter.keys())[0].startswith('module'):
                if not isinstance(self.model,torch.nn.DataParallel):
                    self.model = torch.nn.DataParallel(self.model,device_ids=[0, 1])
                
                self.model.load_state_dict(model_parameter)
            else:
                if isinstance(self.model,torch.nn.DataParallel):
                    self.model.module.load_state_dict(model_parameter)
                else:
                    self.model.load_state_dict(model_parameter)
        else: # cpu
            self.model.load_state_dict(torch.load(
                model_parameter, map_location='cpu'))  # cpu上
        if type=='dev':
            dataset=support_dataset[self.model_name](self.vocab,type,config=self.config)
        if type=='test':
            dataset=support_dataset[self.model_name](self.vocab,type,config=self.config)
        if self.model_name in  ['ggnn','ggnn_simple']:
            self.evaluate_gnn(self.model,dataset,type=type,logger=self.logger,**self.model_config)
        elif self.model_name in ['lstm','gru']:
            self.evaluate_lstm(self.model,dataset,type=type,logger=self.logger,**self.model_config)
        elif self.model_name == 'codebert':
            self.evaluate_codebert(self.model,dataset,type=type,logger=self.logger,**self.model_config)
        elif self.model_name == 'graphcodebert':
            self.evaluate_graphcodebert(self.model,dataset,type=type,logger=self.logger,**self.model_config)
        elif self.model_name == 'astnn':
            self.evaluate_astnn(self.model,dataset,type=type,logger=self.logger,**self.model_config)
        elif self.model_name == 'tbcnn':
            self.evaluate_tbcnn(self.model,dataset,type=type,logger=self.logger,**self.model_config)
    @classmethod
    def train_lstm(cls,model,train_dataset,dev_dataset,logger,**config):
        
        epochs=config['epochs']
        
        train_batch_size=config['train_batch_size']
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,num_workers=4)
        
        optimizer = optimizer_map[config['optimizer']](model.parameters(),lr=float(config['learning_rate']),weight_decay=float(config['l2p']))
       
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=epochs*len(train_dataloader)//config['warmup_steps'],
                                                    num_training_steps=epochs*len(train_dataloader))

        criterion = lossFunc_map[config['loss_func']]()
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Instantaneous batch size per GPU = %d", train_batch_size//max(config['n_gpu'], 1))
        global_step=0 
        train_all_loss=0.0 
        logging_loss=  0.0 
        logging_num=0 
        avg_loss=0.0  
        train_loss=0.0 
        train_num=0 

        
        best_acc=config['save_eval_acc']
        model.zero_grad()
        patiences=config['patiences'] 
        for idx in range(epochs): 
            if patiences==0:
                break
            train_num=0 
            train_loss=0
            bar = tqdm(train_dataloader,total=len(train_dataloader))
            
            for step, batch in enumerate(bar):
                (input_idx,mask,lables)=[x.to(config['device'])  for x in batch]
                model.train()
                
                logits, _ = model(input_idx,mask)
                loss=criterion(logits,lables)
                if config['n_gpu'] > 1:
                    loss = loss.mean()
                if config['gradient_accumulation_steps'] > 1:
                    loss = loss / config['gradient_accumulation_steps']

                
                train_all_loss += loss.item()
                train_num+=1
                train_loss+=loss.item()
                if avg_loss==0:
                    avg_loss=train_all_loss
                    
                avg_loss=round(train_loss/train_num,5)
                bar.set_description("epoch {} loss {}".format(idx,avg_loss))
                loss.backward()
                if (step + 1) % config['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad() 
                    scheduler.step() 
                    global_step += 1
                    if global_step % config['save_steps'] == 0:# 
                        avg_loss=round(np.exp((train_all_loss-logging_loss) /(global_step- logging_num)/config['gradient_accumulation_steps']),4)
                        logger.info(str(global_step)+ "steps"+ " "+"-"*10+"->"+"now train loss is:"+str(avg_loss))
                        results = cls.evaluate_lstm(model=model,eval_dataset=dev_dataset,type='dev',logger=logger,**config)    
                        
                        # Save model checkpoint
                        if results['eval_acc']>best_acc:
                            best_acc=results['eval_acc']
                            logger.info(f"epoch:{idx}, global step:{global_step},{'*'*20}")  
                            logger.info("  Best acc:%s",round(best_acc,4))
                            logger.info("  "+"*"*20)                                   
                            torch.save(model.state_dict(),config['parameter_path'])
                            logger.info("Saving model checkpoint to %s", config['parameter_path'])
                            patiences=config['patiences']
                        else:
                            patiences-=1
                            if patiences==0:
                                logger.warning("continuous without acc improvement, stop traning!")
                                logger.warning("  Best eval acc:%s",round(best_acc,4))
                                break
    @classmethod
    def evaluate_lstm(cls,model,eval_dataset,type='dev',logger=None,**config):
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=config['eval_batch_size'],num_workers=4)

        logger.info(f"***** Running {type} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", config['eval_batch_size'])
        
        
        nb_eval_steps = 0
        model.eval()
        probs=[]  
        y_trues=[]
        for batch in tqdm(eval_dataloader):
            (inputs_ids,mask,labels)=[x.to(config['device'])  for x in batch]
            with torch.no_grad():
                if isinstance(model,torch.nn.DataParallel):
                    prob = model.module.prob(inputs_ids,mask)
                else:
                    prob = model.prob(inputs_ids,mask)
            probs.append(prob.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            nb_eval_steps += 1
        
        #calculate scores
        probs=np.concatenate(probs,0)
        y_trues=np.concatenate(y_trues,0)
        y_preds=np.argmax(probs,axis=1)#   
        
        acc=accuracy_score(y_trues, y_preds)             
        result = {   
            "eval_acc": float(acc)
            
        }

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

        return result
   
   
    @classmethod
    def train_codebert(cls,model,train_dataset,dev_dataset,logger,**config):
       
        epochs=config['epochs']
        
        train_batch_size=config['train_batch_size']
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,num_workers=4)
       
        optimizer = optimizer_map[config['optimizer']](model.parameters(),lr=float(config['learning_rate']),weight_decay=float(config['l2p']))
       
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=epochs*len(train_dataloader)//config['warmup_steps'],
                                                    num_training_steps=epochs*len(train_dataloader))

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Instantaneous batch size per GPU = %d", train_batch_size//max(config['n_gpu'], 1))
        global_step=0 
        train_all_loss=0.0 
        logging_loss=  0.0 
        logging_num=0 
        avg_loss=0.0  
        train_loss=0.0 
        train_num=0 
       
        best_acc=config['save_eval_acc']
        model.zero_grad()
        patiences=config['patiences'] 
        for idx in range(epochs): 
            if patiences==0:
                break
            train_num=0 
            train_loss=0
            bar = tqdm(train_dataloader,total=len(train_dataloader))
            
            for step, batch in enumerate(bar):
                (input_ids,labels)=[x.to(config['device'])  for x in batch]
                model.train()
               
                logits,loss= model(input_ids.squeeze(),labels)# [batch,1,seq_len]=>[batch,seq_len]
                
                if config['n_gpu'] > 1:
                    loss = loss.mean()
                if config['gradient_accumulation_steps'] > 1:
                    loss = loss / config['gradient_accumulation_steps']

                
                train_all_loss += loss.item()
                train_num+=1
                train_loss+=loss.item()
                if avg_loss==0:
                    avg_loss=train_all_loss
                    
                avg_loss=round(train_loss/train_num,5)
                bar.set_description("epoch {} loss {}".format(idx,avg_loss))
                loss.backward() 
                if (step + 1) % config['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad() 
                    scheduler.step() 
                    global_step += 1
                    if global_step % config['save_steps'] == 0:# 
                        avg_loss=round(np.exp((train_all_loss-logging_loss) /(global_step- logging_num)/config['gradient_accumulation_steps']),4)
                        logger.info(str(global_step)+ "steps"+ " "+"-"*10+"->"+"now train loss is:"+str(avg_loss))
                        results = cls.evaluate_codebert(model=model,eval_dataset=dev_dataset,type='dev',logger=logger,**config)    
                        
                        # Save model checkpoint
                        if results['eval_acc']>best_acc:
                            best_acc=results['eval_acc']
                            logger.info(f"epoch:{idx}, global step:{global_step},{'*'*20}")  
                            logger.info("  Best acc:%s",round(best_acc,4))
                            logger.info("  "+"*"*20)                                                 
                            torch.save(model.state_dict(), config['parameter_path'])
                            logger.info("Saving model checkpoint to %s", config['parameter_path'])
                            patiences=config['patiences']
                        else:
                            patiences-=1
                            if patiences==0:
                                logger.warning("continuous without acc improvement, stop traning!")
                                logger.warning("  Best eval acc:%s",round(best_acc,4))
                                break
    @classmethod
    def evaluate_codebert(cls,model,eval_dataset,type='dev',logger=None,**config):
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=config['eval_batch_size'],num_workers=4)

        logger.info(f"***** Running {type} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", config['eval_batch_size'])
        
        
        nb_eval_steps = 0
        model.eval()
        probs=[]  
        y_trues=[]
        for batch in tqdm(eval_dataloader):
            (input_ids,labels)=[x.to(config['device'])  for x in batch]
            with torch.no_grad():
                if isinstance(model,torch.nn.DataParallel):
                    prob = model.module.prob(input_ids.squeeze())
                else:
                    prob = model.prob(input_ids.squeeze())
            probs.append(prob.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            nb_eval_steps += 1
        
        #calculate scores
        probs=np.concatenate(probs,0)
        y_trues=np.concatenate(y_trues,0)
        y_preds=np.argmax(probs,axis=1)#   
        
        acc=accuracy_score(y_trues, y_preds)             
        result = {   
            "eval_acc": float(acc)
            
        }

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

        return result
    
    @classmethod
    def train_graphcodebert(cls,model,train_dataset,dev_dataset,logger,**config):
        
        epochs=config['epochs']
        
        train_batch_size=config['train_batch_size']
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,num_workers=4)
        
        optimizer = optimizer_map[config['optimizer']](model.parameters(),lr=float(config['learning_rate']),weight_decay=float(config['l2p']))
        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=epochs*len(train_dataloader)//config['warmup_steps'],
                                                    num_training_steps=epochs*len(train_dataloader))

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Instantaneous batch size per GPU = %d", train_batch_size//max(config['n_gpu'], 1))
        global_step=0 
        train_all_loss=0.0 
        logging_loss=  0.0 
        logging_num=0 
        avg_loss=0.0  
        train_loss=0.0 
        train_num=0 

        best_acc=config['save_eval_acc']
        model.zero_grad()
        patiences=config['patiences'] 

      
        for idx in range(epochs): 
            if patiences==0:
                break
            train_num=0 
            train_loss=0
            bar = tqdm(train_dataloader,total=len(train_dataloader))
            
            for step, batch in enumerate(bar):
                (inputs_ids,position_idx,attn_mask,lables)=[x.to(config['device'])  for x in batch]
                model.train()
             
                logits,loss= model(inputs_ids,position_idx,attn_mask,lables)
                
                if config['n_gpu'] > 1:
                    loss = loss.mean()
                if config['gradient_accumulation_steps'] > 1:
                    loss = loss / config['gradient_accumulation_steps']

                
                train_all_loss += loss.item()
                train_num+=1
                train_loss+=loss.item()
                if avg_loss==0:
                    avg_loss=train_all_loss
                    
                avg_loss=round(train_loss/train_num,5)
                bar.set_description("epoch {} loss {}".format(idx,avg_loss))
                loss.backward()  
                if (step + 1) % config['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad() 
                    scheduler.step() 
                    global_step += 1
                    if global_step % config['save_steps'] == 0:# 
                        avg_loss=round(np.exp((train_all_loss-logging_loss) /(global_step- logging_num)/config['gradient_accumulation_steps']),4)
                        logger.info(str(global_step)+ "steps"+ " "+"-"*10+"->"+"now train loss is:"+str(avg_loss))
                        results = cls.evaluate_graphcodebert(model=model,eval_dataset=dev_dataset,type='dev',logger=logger,**config)    
                        
                        # Save model checkpoint
                        if results['eval_acc']>best_acc:
                            best_acc=results['eval_acc']
                            logger.info(f"epoch:{idx}, global step:{global_step},{'*'*20}")  
                            logger.info("  Best acc:%s",round(best_acc,4))
                            logger.info("  "+"*"*20)                                                 
                            torch.save(model.state_dict(), config['parameter_path'])
                            logger.info("Saving model checkpoint to %s", config['parameter_path'])
                            patiences=config['patiences']
                        else:
                            patiences-=1
                            if patiences==0:
                                logger.warning("continuous without acc improvement, stop traning!")
                                logger.warning("  Best eval acc:%s",round(best_acc,4))
                                break
    @classmethod
    def evaluate_graphcodebert(cls,model,eval_dataset,type='dev',logger=None,**config):
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=config['eval_batch_size'],num_workers=4)

        logger.info(f"***** Running {type} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", config['eval_batch_size'])
        
        
        nb_eval_steps = 0
        model.eval()
        probs=[]  
        y_trues=[]
        for batch in tqdm(eval_dataloader):
            (inputs_ids,position_idx,attn_mask,labels)=[x.to(config['device'])  for x in batch]
            with torch.no_grad():
                if isinstance(model,torch.nn.DataParallel):
                    prob,lm_loss = model.module.prob(inputs_ids,position_idx,attn_mask,labels)#
                else:
                    prob,lm_loss = model.prob(inputs_ids,position_idx,attn_mask,labels)#
            probs.append(prob.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            nb_eval_steps += 1
        
        #calculate scores
        probs=np.concatenate(probs,0)
        y_trues=np.concatenate(y_trues,0)
        y_preds=np.argmax(probs,axis=1)#   
        
        acc=accuracy_score(y_trues, y_preds)             
        result = {   
            "eval_acc": float(acc)   
        }

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

        return result
    @classmethod
    def train_gnn(cls,model,train_dataset,dev_dataset,logger,**config):
        epochs=config['epochs']
        train_batch_size=config['batch_size']
        
       
        optimizer = optimizer_map[config['optimizer']](model.parameters(),lr=float(config['learning_rate']),weight_decay=float(config['l2p']))
        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=epochs*train_dataset.num_batch//config['warmup_steps'],
                                                    num_training_steps=epochs*train_dataset.num_batch)

        criterion = lossFunc_map[config['loss_func']]()
       
        # Train!
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Instantaneous batch size per GPU = %d", train_batch_size//max(config['n_gpu'], 1))
        
        global_step=0 
        train_all_loss=0.0 
        logging_loss=  0.0 
        logging_num=0 
        avg_loss=0.0  
        train_loss=0.0 
        train_num=0 
        
        best_acc=config['save_eval_acc']

        
        model.zero_grad()
        patiences=config['patiences'] 
        is_first_train=True 
        is_first_eval=True  

        
        for idx in range(epochs): 
            if patiences==0:
                break
            train_num=0 
            train_loss=0
            if idx!=0:
                is_first_train=False
            bar=tqdm(range(train_dataset.num_batch),total=train_dataset.num_batch)
           
            train_dataset.resetBatch()
            for step in bar :
               
                input_batch = train_dataset.nextBatch()            
                input_batch = train_dataset.vectorize_input(input_batch, training=True, device=config['device'],is_first=is_first_train, mode="train")
                """
                input_batch:
                        example = {'batch_size': batch.batch_size,
                        'code_graphs': batch.code_graph,                   
                        'sequences': srcs.to(device) if device else srcs,
                        'sequence_lens': src_lens.to(device) if device else src_lens,
                        'code_token_indexes': batch.code_token_indexes,            
                        'max_code_lens': batch.max_code_length,
                        'labels':labels.to(device) if device else labels
                        }
                """
                    
                model.train()
                
                try:
                    logits= model(input_batch) 
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
                loss=criterion(logits,input_batch['labels'])
                if config['n_gpu'] > 1:
                    loss = loss.mean()
                if config['gradient_accumulation_steps'] > 1:
                    loss = loss / config['gradient_accumulation_steps']
                train_all_loss += loss.item()
                train_num+=1
                train_loss+=loss.item()
                if avg_loss==0:
                    avg_loss=train_all_loss
                    
                avg_loss=round(train_loss/train_num,5)
                bar.set_description("epoch {} loss {}".format(idx,avg_loss))
                loss.backward()
                if (step + 1) % config['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad() 
                    scheduler.step() 
                    global_step += 1 
                    if global_step % config['save_steps'] == 0:
                        avg_loss=round(np.exp((train_all_loss-logging_loss) /(global_step- logging_num)/config['gradient_accumulation_steps']),4)
                        logger.info(f"during the {str(global_step)} steps {'-'*10} ->avg train loss is:{str(avg_loss)}")
                        results = cls.evaluate_gnn(model=model,eval_dataset=dev_dataset,is_first=is_first_eval,logger=logger,**config)
                        is_first_eval=False    
                        
                        # Save model checkpoint
                        if results['eval_acc']>best_acc:
                            best_acc=results['eval_acc']
                            logger.info(f"epoch:{idx}, global step:{global_step},{'*'*20}")   
                            logger.info("  Best acc:%s",round(best_acc,4))
                            logger.info("  "+"*"*20)                          
                                                   
                            torch.save(model.state_dict(), config['parameter_path'])
                            logger.info("Saving model checkpoint to %s", config['parameter_path'])
                            patiences=config['patiences']
                        else:
                            patiences-=1
                            if patiences==0:
                                logger.warning("continuous without acc improvement, stop traning!")
                                logger.warning("  Best eval acc:%s",round(best_acc,4))
                                break
    @classmethod
    def evaluate_gnn(cls,model,eval_dataset,type='dev',is_first=True,logger=None,**config):
        
    
        batch_size=config['eval_batch_size']
        
        
        logger.info(f"***** Running {type} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", batch_size)
        
        
        nb_eval_steps = 0
        model.eval()
        probs=[]  
        y_trues=[]
        bar=tqdm(range(eval_dataset.num_batch),total=eval_dataset.num_batch)
        for step in bar:
            input_batch = eval_dataset.nextBatch()
            
            input_batch = eval_dataset.vectorize_input(input_batch, training=False, device=config['device'],is_first=is_first, mode="eval")
            if input_batch is None: 
                continue
            try:
                with torch.no_grad():
                    if isinstance(model,torch.nn.DataParallel):
                        prob = model.module.prob(input_batch)
                    else:
                        prob = model.prob(input_batch)#
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            probs.append(prob.cpu().numpy())
            y_trues.append(input_batch['labels'].cpu().numpy())
            nb_eval_steps += 1
        
        #calculate scores
        probs=np.concatenate(probs,0)
        y_trues=np.concatenate(y_trues,0)
        y_preds=np.argmax(probs,axis=1) 
        acc=accuracy_score(y_trues, y_preds)             
        result = {   
            "eval_acc": float(acc)
            
        }

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

        return result
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_model',type=str,default='lstm',help="support lstm gru ggnn ggnn_simple")
    parser.add_argument('--train',type=bool,default=True)
    parser.add_argument('--val',type=bool,default=True)
    parser.add_argument('--test',type=bool,default=True)
    parser.add_argument('--pretrain',type=bool,default=False,help='load previous model')
    parser.add_argument('--is_enhance',type=bool,default=False,help='Whether or not to use enhanced data sets for training')
    parser.add_argument('--enhance_type',type=str,default='new_struct',help="gen adv's way")
    parser.add_argument('--enhance_size',type=int,default=-1,help="How much enhanced data to use")
    parser.add_argument('--sample_nums',type=int,default=-1,help="Number of samples sampled from enhanced datasets! = number of enhanced data")
    parser.add_argument('--graph_type',type=str,default='ggnn_bi',help='gcn or ggnn_bi')
    args=parser.parse_args()
    model_name=args.train_model
    if model_name in support_model.keys():
        modehandler=ModelHandler(args)
        modehandler.run()
    else:
        print(f"{model_name} is not yet supported")   