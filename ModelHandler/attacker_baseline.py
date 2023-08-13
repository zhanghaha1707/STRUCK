from datetime import date
import sys
sys.path.append("..")
import torch
from dataset.CodeFeature import CodeFeatures,CodeFeatures_tbcnn
from dataset.GraphDataset import Batch_s
from dataset.vocabClass import VocabModel,VocabModel_tbcnn
from model.LSTM import LSTMClassifier
from model.GRU import GRUClassifier
from utils.basic_setting import *
from utils.tools import *
from utils.pattern import *
import argparse 
from config.support import *
import time
import copy


def get_target_keys(attack_model_name,codefeatures):
        if attack_model_name in ['lstm','gru']:
            keys = list(codefeatures.tokens_with_pos.keys())
        elif attack_model_name in ['tbcnn','astnn']:
            keys=list(codefeatures.tokens_with_pos_all.keys())
        elif attack_model_name  in ['codebert','graphcodebert','ggnn_simple']:
            keys=copy.deepcopy(codefeatures.target_subtokens)
        return keys
def predicte_ggnn_list(model, vocab, codes,label, device):
    codefeature_list = [
        CodeFeatures_ggnn(-1, " ".join(code), label, vocab) for code in codes]
    
    input_batch = GGNNDataset_s.get_model_input(
        codefeature_list, vocab)  
    input_batch = GGNNDataset_s.vectorize_input(
        input_batch, training=False, device=device)
    new_probs = model.prob(input_batch)
   
    new_preds = torch.argmax(new_probs, dim=1)
    return new_preds, new_probs,codefeature_list


def predicte_ggnn(model, vocab,  codefeatures, device):
   
    
    model.to(device)
    input_batch = GGNNDataset_s.get_model_input(
        codefeatures, vocab) 
    input_batch = GGNNDataset_s.vectorize_input(
        input_batch, training=False, device=device)
    label = codefeatures.label
   
    old_probs = model.prob(input_batch)
    old_probs = old_probs.squeeze()
    old_pred = torch.argmax(old_probs)
    if label == old_pred:
        return True, old_pred, old_probs[label]
    else:
        return False, old_pred, old_probs[label]


def predicte_lstm_list(model, vocab, codes,label, device):
    
    codefeature_list=[]
    for code in codes:
        codefeature_list.append(CodeFeatures(-1, " ".join(code), label, vocab))
    
    input_idxs, masks, labels = get_input_from_codefeatures_list(codefeature_list, device)
    new_probs = model.prob(input_idxs, masks)
    new_preds = torch.argmax(new_probs, dim=1)
    return new_preds, new_probs,codefeature_list


def predicte_lstm(model, vocab, codefeatures, device):
    
    model.to(device)
    
    input_idx, mask, label = get_input_from_codefeatures_list(
        [codefeatures], device)
    label = label[0]  
    
    old_probs = model.prob(input_idx, mask).squeeze()
    old_pred = torch.argmax(old_probs)
    if label == old_pred:
        return True, old_pred, old_probs[label]

    else:
        return False, old_pred, old_probs[label]




def predicte_codebert_list(model, vocab, codes,label, device):
    codefeature_list = [
        CodeFeatures_codebert(-1, " ".join(code), label, vocab) for code in codes]
    
    input, _ = CodeBertDataset.get_model_input(codefeature_list)
    input = input.to(device)
    new_probs = model.prob(input)
   
    new_preds = torch.argmax(new_probs, dim=1)
    return new_preds, new_probs,codefeature_list


def predicte_codebert(model, vocab, codefeatures, device):
    # ===
    model.to(device)
    (input_ids, labels) = CodeBertDataset.get_model_input(
        codefeatures)  
    input_ids = input_ids.to(device)
    label = labels[0]  
    
    old_probs = model.prob(input_ids).squeeze()
    old_pred = torch.argmax(old_probs)
    if label == old_pred.item():
        return True, old_pred, old_probs[label]

    else:
        return False, old_pred, old_probs[label]


def predicte_graphcodebert_list(model, vocab, codes,label, device):
    codefeature_list = [
        CodeFeatures_graphcodebert(-1, " ".join(code), label, vocab) for code in codes]
    
    (inputs_ids, position_idx, attn_mask,
     labels) = GraphCodeBertDataset.get_model_input(codefeature_list, device)
    new_probs, _ = model.prob(inputs_ids, position_idx, attn_mask, labels)
    
    new_preds = torch.argmax(new_probs, dim=1)
    return new_preds, new_probs,codefeature_list


def predicte_graphcodebert(model, vocab, codefeatures,  device):
    
    
    model.to(device)
    (inputs_ids, position_idx, attn_mask,
     labels) = GraphCodeBertDataset.get_model_input(codefeatures, device)
    label = labels[0]  
    
    old_probs, _ = model.prob(inputs_ids, position_idx, attn_mask, labels)
    old_probs = old_probs.squeeze()
    old_pred = torch.argmax(old_probs)
    if label == old_pred:
        return True, old_pred, old_probs[label]

    else:
        return False, old_pred, old_probs[label]


predicti_map = {
    'lstm_list': predicte_lstm_list,
    'gru_list': predicte_lstm_list,
    'codebert_list': predicte_codebert_list,
    'graphcodebert_list': predicte_graphcodebert_list,
    'ggnn_simple_list': predicte_ggnn_list,

    'lstm': predicte_lstm,
    'gru': predicte_lstm,
    'codebert': predicte_codebert,
    'graphcodebert': predicte_graphcodebert,
    'ggnn_simple': predicte_ggnn,
}
codefeature_map={
    'lstm': CodeFeatures,
    'gru': CodeFeatures,
    'codebert': CodeFeatures_codebert,
    'graphcodebert': CodeFeatures_graphcodebert,
    'ggnn_simple': CodeFeatures_ggnn, 
}

class StructModify(object):
   
    def __init__(self,model:LSTMClassifier,loss_func,vocab:VocabModel,device:torch.device,**config):        
       
        self.model=model.to(device)
        self.loss_func=loss_func
        self.vocab=vocab
        self.config=config
        self.device=device
        self.random_attack=self.vocab.random_attack
    def insert(self,codefeatures:CodeFeatures,n_candidate=5,change_limits=None,attack_changes=None):     
        
        try:
            pos_candidates = InsAddCandidates(self.attackDict,codefeatures.max_len)
        except:
            pos_candidates = InsAddCandidates(self.attackDict)
        
        pos_nums = len(pos_candidates)
        n_candidate=min(n_candidate,pos_nums)

        candisIdx = random.sample(range(pos_nums), n_candidate)
        
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx] 

     
        can_use_insts=[]
        nums_limits=change_limits-attack_changes
        for inst in self.random_attack:
            if len(inst)<nums_limits:
                can_use_insts.append(inst)
        if len(can_use_insts)==0:
            return [],[]
        new_codes,new_attackDict = [],[]

        for pos in pos_candidates:
            
            
            _attackDict = copy.deepcopy(self.attackDict)
           

            
            inst=random.choice(can_use_insts)
            is_succ=InsAdd(_attackDict, pos, inst)
            if is_succ:
            
                code_raw = InsResult(codefeatures.tokens, _attackDict)
                
                new_codes.append(code_raw)
                new_attackDict.append(_attackDict)
        
        return new_codes,new_attackDict
    def remove(self, codefeatures:CodeFeatures, n_candidate=5):

        pos_candidates = InsDeleteCandidates(self.attackDict) 
        pos_nums = len(pos_candidates)
        n_candidate=min(n_candidate,pos_nums)
        
        candisIdx = random.sample(range(pos_nums), n_candidate)
    
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx]

        new_codes, new_attackDict = [],[]
        for pos, listIdx in pos_candidates:
            _attackDict = copy.deepcopy(self.attackDict)
            InsDelete(_attackDict, pos, listIdx)
            code_raw = InsResult(codefeatures.tokens, _attackDict)
            
            new_codes.append(code_raw)
            new_attackDict.append(_attackDict)

        return new_codes,new_attackDict

    def insert_remove_random(self, codefeatures, n_candidate=10,change_limits=None,attack_changes=None):
        
        new_codes, new_attackDict = [],[]
        fail_cnt = 0
        
        can_use_insts=[]
        nums_limits=change_limits-attack_changes
        for inst in self.random_attack:
            if len(inst)<nums_limits:
                can_use_insts.append(inst)
        while True:
            if fail_cnt >= n_candidate:  
                break
            if random.random() > 0.5: 
                if can_use_insts==[]:
                    fail_cnt += 1
                    continue
                try:
                    pos_candidates = InsAddCandidates(self.attackDict,codefeatures.max_len)
                except:
                    pos_candidates = InsAddCandidates(self.attackDict)
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand = random.sample(pos_candidates, 1)[0]
                
                inst = random.choice(can_use_insts)

                _attackDict = copy.deepcopy(self.attackDict)
                is_succ=InsAdd(_attackDict, pos_cand, inst)
                if not is_succ:
                    continue
            else:
                pos_candidates = InsDeleteCandidates(self.attackDict)
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand, inPosIdx = random.sample(pos_candidates, 1)[0]
                _attackDict = copy.deepcopy(self.attackDict)
                InsDelete(_attackDict, pos_cand, inPosIdx)
            code_raw = InsResult(codefeatures.tokens, _attackDict)
            new_codes.append(code_raw)
            new_attackDict.append(_attackDict)
            break
        return  new_codes,new_attackDict
    def initAttackDict(self, codefeatures:CodeFeatures,mode=0):
       
        self.attackDict={"count":0}
        self.attackDict.update(dict([pos, {'level':indent,'content':[]}] for pos,indent in zip(codefeatures.stmt_attack_poses,codefeatures.stmt_attack_level)))
    @staticmethod
    def statistics_attackDict(attackDict):
        tokens_nums=[]
        levels=[]
        attack_nums=0
        for key,value in attackDict.items():
            if key=='count':
                attack_nums=value
            elif value['content']!=[]:
                tokens_nums.append(0)
                for c in value['content']:
                    tokens_nums[-1]+=len(c)
                levels.append(value['level'])
        return tokens_nums,levels,attack_nums
    @staticmethod
    def counts_attack_dict(attack_dict):
        
        change_nums=0
        for key,item in attack_dict.items():
            if key!="count":
                for add_content in attack_dict[key]['content']:
                    change_nums+=len(add_content)
        return change_nums
class StructAttack(object):
   
    def __init__(self,dataset:LSTMDataset,model:LSTMClassifier,loss_func,vocab:VocabModel,device:torch.device,logger,args,**config) -> None:
   
        self.strucModify=StructModify(model,loss_func,vocab,device,**config)
        self.dataset=dataset
        self.model=model.to(device)
        self.attack_model_name = args.attack_model
        if isinstance(self.model,torch.nn.DataParallel):
            self.model=self.model.module
        self.vocab=vocab
        self.logger=logger
        self.device=device
        self.enhance_data=[]
        self.args=args
        self.config=config
        
        self.succ_info={'tokens_nums':[],"attack_nums":[],"attack_levels":[]}
        try:
            logger.warning(f"STRUCT ATTACK:{args.attack_way} attack for Model:{args.attack_model} DATASET:{args.attack_data} information is saved in {config['log_path']}")
        except:
            pass
        myConfig.show_config(self.config,name='StructAttack',logger=self.logger)
    def attack_single_code(self,codefeatures:CodeFeatures,is_log=True):
        
        is_right, old_pred, old_prob = predicti_map[self.attack_model_name](self.model, self.vocab, codefeatures,self.device)
        
    
        
        if not is_right:
            if is_log:
                self.logger.info("SUCC! Original mistake.")
            return True,codefeatures,old_pred,0,None,None,None 
        correct_prob=old_prob
       
        codefeatures.prepare_for_struct_attack()
        self.strucModify.initAttackDict(codefeatures)
        n_candidate= self.config['n_candidate']
        n_iter=self.config['n_iter'] 
        attack_times=0 
        continuous_fails=0 
        patience=len(codefeatures.stmt_attack_poses)
        
        change_limits=len(codefeatures.tokens)*self.config['change_limts_factor']
        attack_changes=counts_old_attack_dict(self.strucModify.attackDict)
        invotions=0
        gen_adv_nums=0
        while attack_times<n_iter:
            attack_times+=1
           
            if len(self.strucModify.attackDict.keys())==1:
                if is_log:
                    self.logger.info(f"this code without poses can be structed attack\n")
                break
            if self.args.attack_way=="insert":
                
                n_could_del = self.strucModify.attackDict["count"]
                n_candidate_del = n_could_del
                n_candidate_ins = n_candidate - n_candidate_del
                assert n_candidate_del >= 0 and n_candidate_ins >= 0 
                                      
                new_codes_del,new_attackDicts_del = self.strucModify.remove(codefeatures, n_candidate_del)
                new_codes_add,new_attackDicts_add = self.strucModify.insert(codefeatures, n_candidate_ins,change_limits,attack_changes=attack_changes)
                new_codes = new_codes_del + new_codes_add
                new_attackDicts = new_attackDicts_del + new_attackDicts_add
            elif self.args.attack_way=="random_insert":
                new_codes,new_attackDicts=self.strucModify.insert_remove_random(codefeatures, n_candidate,change_limits,attack_changes=attack_changes)
            if new_attackDicts == []:            
                continuous_fails += 1
                continue
            
            new_preds, new_probs,candi_codefeatures = predicti_map[self.attack_model_name+'_list'](
                    self.model, self.vocab, new_codes,codefeatures.label, self.device)
            invotions+=1
            gen_adv_nums+=len(new_attackDicts)
            for candi_codefeature,new_attackDict,new_prob,new_pred in zip(candi_codefeatures,new_attackDicts,new_probs,new_preds):
                if new_pred!=old_pred:
                    attack_changes=counts_old_attack_dict(self.strucModify.attackDict)
                    
                    tokens_nums,levels,attack_nums=self.strucModify.statistics_attackDict(new_attackDict)
                    if is_log:
                        self.logger.info(f"SUCC!\t insert_nums => {attack_nums}\t\t invotions=>{invotions}\t\t gen_adv_nums=>{gen_adv_nums}\t\t attack_changes=>{attack_changes}\t\t{old_pred.item()}:{correct_prob.item():4f} => {new_pred.item()}: {new_prob[new_pred].item():4f}\n" )
                    
                    return True,candi_codefeature,new_pred,1,invotions,gen_adv_nums,attack_changes
           
            min_correct_prob_idx = torch.argmin(new_probs[:, old_pred])
            if new_probs[min_correct_prob_idx][old_pred] < correct_prob:
                self.strucModify.attackDict = new_attackDicts[min_correct_prob_idx]
                
                attack_changes=counts_old_attack_dict(self.strucModify.attackDict)
                continuous_fails = 0
                
                tokens_nums,levels,insert_nums=self.strucModify.statistics_attackDict(new_attackDicts[min_correct_prob_idx])
                if is_log:
                    self.logger.info(f"acc\t insert_nums => {insert_nums}\t\t{old_pred.item()}({correct_prob.item():4f}) => {old_pred.item()}({new_probs[min_correct_prob_idx][old_pred].item():4f})\n" )
                
                correct_prob=new_probs[min_correct_prob_idx][old_pred]
            
            else:
                continuous_fails += 1 
                if is_log:
                    self.logger.info(f"rej\t this struct attack after {self.strucModify.attackDict['count']} attacks\n")
            if continuous_fails>patience:
                break   
        if is_log:
            self.logger.info("FAIL!")
        return False, codefeatures, old_pred,2,None,None,None

    def attack_dataset(self):
        
        is_enhance_data=self.config['is_enhance_data']

        succ_times=self.config["last_succ_times"] 
        total_time=self.config["last_total_time"] 
        total_invotions=self.config['last_total_invotions']
        total_advs_nums=self.config['last_advs_nums']
        total_changes_nums=self.config['last_total_changes_nums']
        data=self.dataset.data
        data_len=len(data)
        st_time=time.time()
        last_code_idx=self.config['last_attack_code_idx']
        attack_codes_nums=self.config['last_attack_codes_nums']
        if args.sample_nums >0 and args.attack_data=='train':
            
            targt_dates=random.sample(data[last_code_idx:],args.sample_nums) 
        else:
            targt_dates=data[last_code_idx:]
        tbar = tqdm(targt_dates, file=sys.stdout)
        fail_pred_num=0
        for i, target_codefeatures in enumerate(tbar):
            
            self.logger.info(f"\t{i+last_code_idx+1}/{data_len}\t ID = {target_codefeatures.id}\tY = {target_codefeatures.label}")
            start_time = time.time()
            try:
               
                is_succ, adv_codefeatures, adv_label,result_type,invotions,gen_adv_nums,changes_nums=self.attack_single_code(target_codefeatures)
                if is_succ:
                    fail_pred_num+=1
                if result_type==1:
                    succ_times+=1
                    total_time+=time.time()-start_time
                    total_invotions+=invotions
                    total_advs_nums+=gen_adv_nums
                    total_changes_nums+=changes_nums
                    if is_enhance_data and result_type==1:
                        self.enhance_data.append(adv_codefeatures)
                if result_type in [1,2]:
                    attack_codes_nums+=1  
                
                if succ_times:
                    self.logger.info(f"Curr succ rate ={succ_times}/{(attack_codes_nums)}=>{succ_times/(attack_codes_nums):3f}, Avg time cost ={total_time:1f}/{succ_times}=>{total_time/succ_times:1f},Avg invo={total_invotions}/{succ_times}=>{total_invotions/succ_times:1f},Avg gen_advs={total_advs_nums}/{succ_times}=>{total_advs_nums/succ_times:1f}, Adv changes nums={total_changes_nums}/{succ_times}=>{total_changes_nums/succ_times:1f} \n")
                else:
                    self.logger.info(f"Curr succ rate = 0, Avg time cost = NaN sec, Avg invo = NaN\n")
            except:
                pass
                # file = self.args.error_log_path
                # with open(file, 'a', encoding='utf-8') as fileobj:
                #     fileobj.write(str(i+last_code_idx+1)+'\n')
            tbar.set_description(f"Curr succ rate {succ_times/(attack_codes_nums):3f}")  
        self.logger.info(f"[Task Done] Time Cost: {time.time()-st_time:1f} sec Succ Rate: {succ_times/attack_codes_nums:.3f}\n")
        if is_enhance_data:
            self.logger.info ("Enhance data Number: %d (Out of %d False Predicted data)" % (fail_pred_num, len(self.enhance_data)))
            if args.sample_nums <=0:
                save_path=os.path.join(self.config['adv_data_path'],self.args.attack_model,self.args.attack_data+"_"+self.args.attack_way+".pkl")
            else:
                save_path=os.path.join(self.config['adv_data_path'],self.args.attack_model,self.args.attack_data+"_"+self.args.attack_way+"_"+str(args.sample_nums)+".pkl")
            if len(self.enhance_data)>0:
                save_pkl(self.enhance_data,save_path)
                self.logger.warning(f"enhanced data is saved in {save_path}")
            else:
                self.logger.warning(f"Failed to enhance data.0/{args.sample_nums}")
class TokenRename(object):
   
    def __init__(self,model:LSTMClassifier,loss_func,vocab:VocabModel,device:torch.device,**config):        
       
        self.model=model.to(device)
        self.loss_func=loss_func
        self.vocab=vocab
        self.config=config
        self.device=device
        self.token_mask=self.get_token_mask_on_vocab()
    def get_token_mask_on_vocab(self):
       

        mask=torch.zeros(self.model.vocab_size)
        mask.index_put_([torch.LongTensor(self.vocab.all_candidate_idxs)], torch.Tensor([1 for _ in self.vocab.all_candidate_idxs]))
        mask = mask.reshape([self.model.vocab_size, 1]).to(self.device) 
        return mask
    def get_token_mask_on_seq(self,target_token_poses):
  
        mask=torch.zeros(self.model.max_len)
        mask.index_put_([torch.LongTensor(target_token_poses)], torch.Tensor([1 for _ in target_token_poses]))
        mask = mask.reshape([self.model.max_len, 1]).to(self.device)
        return mask
    def get_token_mask_for_code(self,tokens):
        
        token_mask=copy.deepcopy(self.token_mask)
        token_idxs=self.vocab.tokenize(tokens)
        for idx in token_idxs:
            token_mask[idx]=0
        return token_mask
    def get_new_code(self,codefeatures:CodeFeatures,target_token,rename_token_idx):
        
        
        rename_token=self.vocab[rename_token_idx]
        tokens=codefeatures.tokens
        for pos in codefeatures.tokens_with_pos_all[target_token]:
            tokens[pos]=rename_token
      
        return tokens
    def get_codebert_new_code(self,codefeatures:CodeFeatures_codebert,target_token,rename_token_idx):
        rename_token=self.vocab[rename_token_idx]
        if rename_token.startswith('Ġ'):
            rename_token=rename_token[1:]
        if target_token.startswith('Ġ'):
            target_token=target_token[1:]
        tokens=codefeatures.tokens
        new_tokens=[]
        for token in tokens:
            new_token=token
            clear_token=self.vocab.getToken(token)
            if  clear_token not in self.vocab.forbidden_tokens and not clear_token.startswith('<'):
                if target_token in token :
                    temp_token=token.replace(target_token,rename_token)
                    if temp_token not in self.vocab.forbidden_tokens and self.vocab[temp_token] not in self.vocab.tokenizer.all_special_ids:
                        new_token=token.replace(target_token,rename_token)  
            new_tokens.append(new_token)
        return new_tokens
    def get_ggnn_new_code(self,codefeatures:CodeFeatures_ggnn,target_token,rename_token_idx):
        
        
        rename_token=self.vocab.trnas_node(rename_token_idx)
        tokens=codefeatures.tokens
        new_tokens=[]
        node_changes=0
        for token in tokens:
            new_token=token
            clear_token=self.vocab.getToken(token)
            if  clear_token not in self.vocab.forbidden_tokens and not clear_token.startswith('<'):
                if target_token in token :
                    temp_token=token.replace(target_token,rename_token)
                    if temp_token not in self.vocab.forbidden_tokens and self.vocab.trnas_node(temp_token)!=self.vocab.UNK:
                        new_token=token.replace(target_token,rename_token)  
                        node_changes+=1
            new_tokens.append(new_token)
        return new_tokens
    def random_rename(self,codefeatures:CodeFeatures,target_token:str,all_cand_tokens:list,n_candidate=5):    
       
        
        rename_token_cand=random.sample(all_cand_tokens,n_candidate)
        
        rename_token_cand_idx=self.vocab.tokenize(rename_token_cand)
        new_codes = []
        new_tokens_idx=[]
        for rename_token_idx in rename_token_cand_idx:
            if self.vocab[rename_token_idx] in codefeatures.tokens_with_pos_all.keys():
                continue
           
            new_codes.append(self.get_new_code(copy.deepcopy(codefeatures),target_token,rename_token_idx))
            new_tokens_idx.append(rename_token_idx)
            if len(new_codes)==n_candidate:
                break
      
        return new_codes, new_tokens_idx
    def random_rename_for_codebert(self,codefeatures:CodeFeatures_codebert,target_token:str,all_cand_tokens:list,n_candidate=5):
        rename_token_cand=random.sample(all_cand_tokens,n_candidate)
        rename_token_cand_idx=[self.vocab[rename_token] for rename_token in rename_token_cand]
        new_codes = []
        new_tokens_idx=[]
        for rename_token_idx in rename_token_cand_idx:
            
            new_codes.append(self.get_codebert_new_code(copy.deepcopy(codefeatures),target_token,rename_token_idx))
            new_tokens_idx.append(rename_token_idx)
        return new_codes,new_tokens_idx
    def random_rename_for_ggnn(self,codefeatures:CodeFeatures_ggnn,target_token:str,all_cand_tokens:list,n_candidate=5):
        rename_token_cand=random.sample(all_cand_tokens,len(all_cand_tokens))
        rename_token_cand_idx=[self.vocab.trnas_node(rename_token) for rename_token in rename_token_cand]
        new_codes = []
        new_tokens_idx=[]
        for rename_token_idx in rename_token_cand_idx :
            if self.token_mask[rename_token_idx]:
               
                new_code=self.get_ggnn_new_code(copy.deepcopy(codefeatures),target_token,rename_token_idx)
                new_codes.append(new_code)
                new_tokens_idx.append(rename_token_idx)
                if len(new_codes)==n_candidate:
                    break
        return new_codes, new_tokens_idx
class RenameAttack(object):
    
    def __init__(self,dataset:LSTMDataset,model:LSTMClassifier,loss_func,vocab:VocabModel,device:torch.device,logger,args,**config) -> None:
        
        self.tokenRename=TokenRename(model,loss_func,vocab,device,**config)
        self.dataset=dataset
        self.attack_model_name = args.attack_model
        self.model=model.to(device)
        if isinstance(self.model,torch.nn.DataParallel):
            self.model=self.model.module
        self.vocab=vocab
        self.logger=logger
        self.device=device
        self.enhance_data=[]
        self.args=args
        self.config=config
        logger.warning(f"RENAME:{args.attack_way} attack for Model:{args.attack_model} DATASET:{args.attack_data} information is saved in {config['log_path']}")
        myConfig.show_config(self.config,name='renameAttack',logger=self.logger)
    def attack_dataset(self):
        
        
        is_enhance_data=self.config['is_enhance_data']

        succ_times=self.config["last_succ_times"] 
        total_time=self.config["last_total_time"] 
        total_invotions=self.config['last_total_invotions']
        total_advs_nums=self.config['last_advs_nums']
        total_changes_nums=self.config['last_total_changes_nums']
        data=self.dataset.data
        data_len=len(data)
        st_time=time.time()
        last_code_idx=self.config['last_attack_code_idx']
        attack_codes_nums=self.config['last_attack_codes_nums']# 
        if args.sample_nums >0:
            
            targt_dates=random.sample(data[last_code_idx:],args.sample_nums) 
        else:
            targt_dates=data[last_code_idx:]
        tbar = tqdm(targt_dates, file=sys.stdout)
        fail_pred_num=0
        
        for i, target_codefeatures in enumerate(tbar):
            
            self.logger.info(f"\t{i+last_code_idx+1}/{data_len}\t ID = {target_codefeatures.id}\tY = {target_codefeatures.label}")
            start_time = time.time()
            try:
                
                is_succ, adv_codefeatures, adv_label,result_type,invotions,gen_adv_nums,changes_nums=self.attack_single_code(target_codefeatures)
                
                if is_succ:
                        fail_pred_num+=1
                if result_type==1:
                    succ_times+=1
                    total_time+=time.time()-start_time
                    total_invotions+=invotions
                    total_advs_nums+=gen_adv_nums
                    total_changes_nums+=changes_nums
                    if is_enhance_data and result_type==1:
                        self.enhance_data.append(adv_codefeatures)
                if result_type in [1,2]:
                    attack_codes_nums+=1    
               
                if succ_times:
                    self.logger.info(f"Curr succ rate ={succ_times}/{(attack_codes_nums)}=>{succ_times/(attack_codes_nums):3f}, Avg time cost ={total_time:1f}/{succ_times}=>{total_time/succ_times:1f},Avg invo={total_invotions}/{succ_times}=>{total_invotions/succ_times:1f},Avg gen_advs={total_advs_nums}/{succ_times}=>{total_advs_nums/succ_times:1f}, Adv changes nums={total_changes_nums}/{succ_times}=>{total_changes_nums/succ_times:1f} \n")
                else:
                    self.logger.info(f"Curr succ rate = 0, Avg time cost = NaN sec, Avg invo = NaN\n")
            except:
                pass
                # file = self.args.error_log_path
                # with open(file, 'a', encoding='utf-8') as fileobj:
                #     fileobj.write(str(i+last_code_idx+1)+'\n')
            tbar.set_description(f"Curr succ rate {succ_times/(attack_codes_nums):3f}")  
        self.logger.info(f"[Task Done] Time Cost: {time.time()-st_time:1f} sec Succ Rate: {succ_times/attack_codes_nums:.3f}\n")
        if is_enhance_data:
            self.logger.info("Enhance data Number: %d (Out of %d False Predicted data)" % (fail_pred_num, len(self.enhance_data)))
            if args.sample_nums <=0:
                save_path=os.path.join(self.config['adv_data_path'],self.args.attack_model,self.args.attack_data+"_"+self.args.attack_way+".pkl")
            else:
                save_path=os.path.join(self.config['adv_data_path'],self.args.attack_model,self.args.attack_data+"_"+self.args.attack_way+"_"+str(args.sample_nums)+".pkl")
            if len(self.enhance_data)>0:
                save_pkl(self.enhance_data,save_path)
                self.logger.warning(f"enhanced data is saved in {save_path}")
            else:
                self.logger.warning(f"Failed to enhance data.0/{args.sample_nums}")  
    def attack_single_code(self,codefeatures:CodeFeatures):
       
        

       
        is_right, old_pred, old_prob = predicti_map[self.attack_model_name](self.model, self.vocab, codefeatures,self.device)
        
        if not is_right:
            self.logger.info("SUCC! Original mistake.")
            return True,codefeatures,old_pred,0,None,None,None 
        correct_prob=old_prob
        
       

        codefeatures.prepare_for_rename(self.vocab)
        n_candidate= self.config['n_candidate']
        n_iter=self.config['n_iter'] 
        attack_times=0 
        continuous_fails=0        
        invotions=0
        gen_adv_nums=0
        
        keys=self.get_target_keys(codefeatures)
        all_cand_tokens=[token for token in self.vocab.all_candidate_tokens if token not in keys ]
        save_code_tokens=copy.deepcopy(codefeatures.tokens)
        while attack_times<n_iter:
            
            if len(keys)==0:
                self.logger.info(f"this code without tokens can be renamed\n")
                break
            for target_token in keys:
                if attack_times>=n_iter:
                    
                    break
                if continuous_fails>=len(keys):
                    attack_times=n_iter
                    break
                attack_times+=1 
                
                if self.args.attack_way=="random_rename" and self.attack_model_name in ['lstm','gru','tbcnn','astnn']:
                    new_codes, rename_token_cand_idx=self.tokenRename.random_rename(codefeatures,target_token,all_cand_tokens,n_candidate)
                elif self.args.attack_way=="random_rename" and self.attack_model_name in ['codebert','graphcodebert']:
                    new_codes, rename_token_cand_idx=self.tokenRename.random_rename_for_codebert(codefeatures,target_token,all_cand_tokens,n_candidate)
                elif self.args.attack_way=="random_rename" and self.attack_model_name in ['ggnn_simple']:
                    new_codes, rename_token_cand_idx=self.tokenRename.random_rename_for_ggnn(codefeatures,target_token,all_cand_tokens,n_candidate)
                if len(new_codes)==0:
                    self.logger.info(f"rej\t {target_token}")
                    continuous_fails += 1
                    continue
                
                new_preds, new_probs,candi_codefeatures = predicti_map[self.attack_model_name+'_list'](
                    self.model, self.vocab, new_codes,codefeatures.label, self.device)
                invotions+=1
                gen_adv_nums+=len(new_codes)
                

                for candi_codefeature,rename_token_idx, new_prob,new_pred in zip(candi_codefeatures,rename_token_cand_idx,new_probs,new_preds):
                    if new_pred!=old_pred:
                       
                        changes_nums=counts_rename_changes(save_code_tokens,candi_codefeature.tokens)
                        if self.attack_model_name in ['ggnn_simple']:
                            self.logger.info(f"SUCC!\t {target_token} => {self.vocab.trnas_node(rename_token_idx)}\t\t {old_pred.item()}:{correct_prob.item():2f} => {new_pred.item()}: {new_prob[new_pred].item():2f} \t\t changes_nums=>{changes_nums}\n" )
                        else:
                            self.logger.info(f"SUCC!\t {target_token} => {self.vocab[rename_token_idx]}\t\t {old_pred.item()}:{correct_prob.item():2f} => {new_pred.item()}: {new_prob[new_pred].item():2f} \t\t changes_nums=>{changes_nums}\n" )
                        #
                        return True,candi_codefeature,new_pred,1,invotions,gen_adv_nums,changes_nums
                
                min_correct_prob_idx = torch.argmin(new_probs[:, old_pred])
                if new_probs[min_correct_prob_idx][old_pred] < correct_prob:
                    codefeatures = candi_codefeatures[min_correct_prob_idx]
                    codefeatures.prepare_for_rename(self.vocab)
                    
                    if self.attack_model_name in ['ggnn_simple']:
                        all_cand_tokens.remove(self.vocab.trnas_node(rename_token_cand_idx[min_correct_prob_idx]))
                    else:
                        all_cand_tokens.remove(self.vocab[rename_token_cand_idx[min_correct_prob_idx]])
                    all_cand_tokens.append(target_token)
                    continuous_fails = 0
                    if self.attack_model_name in ['ggnn_simple']:
                        self.logger.info(f"acc\t {target_token} => {self.vocab.trnas_node(rename_token_cand_idx[min_correct_prob_idx])}\t\t{old_pred.item()}({correct_prob.item():4f}) => {old_pred.item()}({new_probs[min_correct_prob_idx][old_pred].item():4f})\n" )
                    else:
                        self.logger.info(f"acc\t {target_token} => {self.vocab[rename_token_cand_idx[min_correct_prob_idx]]}\t\t{old_pred.item()}({correct_prob.item():4f}) => {old_pred.item()}({new_probs[min_correct_prob_idx][old_pred].item():4f})\n" )
                    correct_prob=new_probs[min_correct_prob_idx][old_pred]
                
                else:
                    continuous_fails += 1 
                    self.logger.info(f"rej\t {target_token}\n")
            keys=self.get_target_keys(codefeatures)     
        self.logger.info("FAIL!")
        return False, codefeatures, None,2,None,None,None
    def get_target_keys(self,codefeatures):
        if self.attack_model_name in ['lstm','gru']:
            keys = list(codefeatures.tokens_with_pos.keys())
        elif self.attack_model_name in ['tbcnn','astnn']:
            keys=list(codefeatures.tokens_with_pos_all.keys())
        elif self.attack_model_name  in ['codebert','graphcodebert','ggnn_simple']:
            keys=copy.deepcopy(codefeatures.target_subtokens)
        return keys

class MHMAttack(object):
   
    def __init__(self,dataset:LSTMDataset,model:LSTMClassifier,vocab:VocabModel,device:torch.device,logger,args,**config):
        self.dataset=dataset
        self.args=args
        self.model=model.to(device)
        self.attack_model_name = args.attack_model
        if isinstance(self.model,torch.nn.DataParallel):
            self.model=self.model.module
        self.vocab=vocab
        self.config=config
        self.device=device
        self.enhance_data=[]  
        self.token_mask=self.get_token_mask_on_vocab()
        self.logger=logger
        logger.warning(f"RENAME:{args.attack_way} attack for Model:{args.attack_model} DATASET:{args.attack_data} information is saved in {config['log_path']}")
        myConfig.show_config(self.config,name='MHMAttack',logger=self.logger)
    def get_token_mask_on_vocab(self):
       
        
        mask=torch.zeros(self.model.vocab_size)# vacab中那些token是可以用的
        mask.index_put_([torch.LongTensor(self.vocab.all_candidate_idxs)], torch.Tensor([1 for _ in self.vocab.all_candidate_idxs]))# 能够有效替换的token为1,size:[vocab_size]
        mask = mask.reshape([self.model.vocab_size, 1]).to(self.device) # [vacab_size*1]0，1向量
        return mask
    def get_token_mask_for_code(self,tokens):
        
        token_mask=copy.deepcopy(self.token_mask)
        token_idxs=self.vocab.tokenize(tokens)
        for idx in token_idxs:
            token_mask[idx]=0
        return token_mask
    def mcmc(self,codefeatures:CodeFeatures):
        
        is_right, old_pred, old_prob = predicti_map[self.attack_model_name](self.model, self.vocab, codefeatures,self.device)
        
        if not is_right:
            self.logger.info("SUCC! Original mistake.")
            return True,codefeatures,old_pred,0,None,None,None 
        
       
        codefeatures.prepare_for_rename(self.vocab)
        invotions=0
        gen_adv_nums=0
        n_candidate=self.config['n_candidate']
        max_iter=self.config['n_iter']
        prob_threshold=self.config['prob_threshold']
        candi_mode=self.config['candidate_mode'] 
       
        keys=get_target_keys(self.attack_model_name,codefeatures)
        all_cand_tokens=[token for token in self.vocab.all_candidate_tokens if token not in keys ]
        save_code_tokens=copy.deepcopy(codefeatures.tokens) 
        for iteration in range(1,1+max_iter):
            if len(keys)==0:
                self.logger.info(f"this code without tokens can be renamed\n")
                break

            res=self.random_attack_selected_token(codefeatures,keys,n_candidate,prob_threshold,candi_mode,all_cand_tokens)
            invotions+=1
            gen_adv_nums+=res['gen_adv_nums']
            self.log_result(iteration,res)
            
            if res['status'].lower() == 's':
                changes_nums=counts_rename_changes(save_code_tokens,res['code'].tokens)
                return True, res['code'],None,1,invotions,gen_adv_nums,changes_nums
            if res['status'].lower()=='a':
                all_cand_tokens.append(res['target_token'])
                all_cand_tokens.remove(res['rename_token'])
                codefeatures=res['code']
                codefeatures.prepare_for_rename(self.vocab)
                keys=get_target_keys(self.attack_model_name,codefeatures)
        self.logger.info("FAIL!")
        return False, None,None,2,None,None,None
    def random_attack_selected_token(self,codefeatures:CodeFeatures,keys=None,n_candidate=30, prob_threshold=0.95, candi_mode="random",all_cand_tokens=None):
        assert candi_mode.lower() in ["random", "nearby"]
        selected_token = random.sample(keys,1)[0]
        label=codefeatures.label
        if candi_mode == "random":
            # First, generate candidate set.
            # The transition probabilities of all candidate are the same.
            candi_token = [selected_token]
            candi_tokens = [copy.deepcopy(codefeatures.tokens)]
            
            if all_cand_tokens is None:
                all_cand_tokens=[token for token in self.vocab.all_candidate_tokens if token not in keys ]
            rename_token_cand=random.sample(all_cand_tokens,n_candidate)
            for rename_token in rename_token_cand:
                if rename_token in keys:
                    continue
                     
                candi_token.append(rename_token)
                
                if self.attack_model_name in ['lstm','gru','tbcnn','astnn']:
                    candi_tokens.append(self.get_new_code(copy.deepcopy(codefeatures),selected_token,self.vocab[rename_token]))
                elif self.attack_model_name in ['codebert','graphcodebert']:
                    candi_tokens.append(self.get_codebert_new_code(copy.deepcopy(codefeatures),selected_token,self.vocab[rename_token]))
                elif self.attack_model_name in ['ggnn_simple']:
                    candi_tokens.append(self.get_ggnn_new_code(copy.deepcopy(codefeatures),selected_token,self.vocab.trnas_node(rename_token)))
            
            new_preds, new_probs,candi_codefeatures = predicti_map[self.attack_model_name+'_list'](
                self.model, self.vocab, candi_tokens,label, self.device)
            gen_adv_nums=len(candi_token)-1
            
            
            for i in range(1,len(candi_token)):   # Find a valid example
                if new_preds[i] != label:
                    return {"status": "s", "alpha": 1, "code": candi_codefeatures[i],
                            "target_token": selected_token, "rename_token": candi_token[i],
                            "old_prob": new_probs[0,label], "new_prob": new_probs[i,label],
                            "old_pred": new_preds[0], "new_pred": new_preds[i],
                            'gen_adv_nums':gen_adv_nums}
            
            candi_idx = 0
            min_prob = new_probs[0,label]

            for idx, a_prob in enumerate(new_probs[1:]):
                if a_prob[label] < min_prob:
                    candi_idx = idx + 1
                    min_prob = a_prob[label]
            
            # At last, compute acceptance rate.
            alpha = (1-new_probs[candi_idx][label]+1e-10) / (1-new_probs[0][label]+1e-10) 
            
            if random.uniform(0, 1) > alpha or alpha < prob_threshold or candi_idx==0:
                return {"status": "r", "alpha": alpha, "code": candi_codefeatures[candi_idx],
                        "target_token": selected_token, "rename_token": candi_token[candi_idx],
                        "old_prob": new_probs[0][label], "new_prob": new_probs[candi_idx][label],
                        "old_pred": new_preds[0], "new_pred": new_preds[candi_idx],
                        'gen_adv_nums':gen_adv_nums
                        }
            else:
                return {"status": "a", "alpha": alpha, "code": candi_codefeatures[candi_idx],
                        "target_token": selected_token, "rename_token": candi_token[candi_idx],
                        "old_prob": new_probs[0][label], "new_prob": new_probs[candi_idx][label],
                        "old_pred": new_preds[0], "new_pred": new_preds[candi_idx],
                        'gen_adv_nums':gen_adv_nums}
        else:
            pass
    
    def get_new_code(self,codefeatures:CodeFeatures,target_token,rename_token_idx):
       
        
        rename_token=self.vocab[rename_token_idx]
        tokens=codefeatures.tokens
        for pos in codefeatures.tokens_with_pos_all[target_token]:
            tokens[pos]=rename_token
       
        return tokens
    def get_codebert_new_code(self,codefeatures:CodeFeatures_codebert,target_token,rename_token_idx):
        
        
        rename_token=self.vocab[rename_token_idx]
        if rename_token.startswith('Ġ'):
            rename_token=rename_token[1:]
        if target_token.startswith('Ġ'):
            target_token=target_token[1:]
        tokens=codefeatures.tokens
        new_tokens=[]
        for token in tokens:
            new_token=token
            clear_token=self.vocab.getToken(token)
            if  clear_token not in self.vocab.forbidden_tokens and not clear_token.startswith('<'):
                if target_token in token :
                    temp_token=token.replace(target_token,rename_token)
                    if temp_token not in self.vocab.forbidden_tokens and self.vocab[temp_token] not in self.vocab.tokenizer.all_special_ids:
                        new_token=token.replace(target_token,rename_token)  
            new_tokens.append(new_token)
        return new_tokens
    def get_ggnn_new_code(self,codefeatures:CodeFeatures_ggnn,target_token,rename_token_idx):
       
        
        rename_token=self.vocab.trnas_node(rename_token_idx)
        tokens=codefeatures.tokens
        new_tokens=[]
        for token in tokens:
            new_token=token
            clear_token=self.vocab.getToken(token)
            if  clear_token not in self.vocab.forbidden_tokens and not clear_token.startswith('<'):
                if target_token in token :
                    temp_token=token.replace(target_token,rename_token)
                    if temp_token not in self.vocab.forbidden_tokens and self.vocab.trnas_node(temp_token)!=self.vocab.UNK:
                        new_token=token.replace(target_token,rename_token)  
            new_tokens.append(new_token)
        return new_tokens
    def log_result(self,iter=None,res=None):
        if res['status'].lower() == 's':
                self.logger.info(f"SUCC!\t {res['target_token']} => {res['rename_token']}\t\t {res['old_pred']}:{res['old_prob']} => {res['new_pred']}: {res['new_prob']:4f}\t a={res['alpha']:3f}\n" )
        elif res['status'].lower() == 'r': # Rejected
            self.logger.info(f"rej\t {res['target_token']}")
            # self.logger.info(f"rej\t {res['target_token']} => {res['rename_token']}\t\t {res['old_pred']}:{res['old_prob']} => {res['new_pred']}: {res['new_prob']:4f}\t a={res['alpha']:3f}\n" )
        elif res['status'].lower() == 'a': # Accepted
            self.logger.info(f"acc\t {res['target_token']} => {res['rename_token']}\t\t {res['old_pred']}:{res['old_prob']} => {res['new_pred']}: {res['new_prob']:4f}\t a={res['alpha']:3f}\n" )
    def attack_dataset(self):
       
        
        is_enhance_data=self.config['is_enhance_data']

        succ_times=self.config["last_succ_times"] 
        total_time=self.config["last_total_time"] 
        total_invotions=self.config['last_total_invotions']
        total_advs_nums=self.config['last_advs_nums']
        total_changes_nums=self.config['last_total_changes_nums']
        data=self.dataset.data
        data_len=len(data)
        st_time=time.time()
        last_code_idx=self.config['last_attack_code_idx']
        attack_codes_nums=self.config['last_attack_codes_nums']
        if args.sample_nums >0:
            
            targt_dates=random.sample(data[last_code_idx:],args.sample_nums) 
        else:
            targt_dates=data[last_code_idx:]
        tbar = tqdm(targt_dates, file=sys.stdout)
        fail_pred_num=0
        
        for i, target_codefeatures in enumerate(tbar):
            
            
            self.logger.info(f"\t{i+last_code_idx+1}/{data_len}\t ID = {target_codefeatures.id}\tY = {target_codefeatures.label}")
            start_time = time.time()
            
            try:
                is_succ, adv_codefeatures,new_pred,result_type,invotions,gen_adv_nums,changes_nums=self.mcmc(target_codefeatures)
                if is_succ:
                    fail_pred_num+=1
                if result_type==1:
                    succ_times+=1
                    total_time+=time.time()-start_time
                    total_invotions+=invotions
                    total_advs_nums+=gen_adv_nums
                    total_changes_nums+=changes_nums
                    if is_enhance_data and result_type==1:
                        self.enhance_data.append(adv_codefeatures)
                if result_type in [1,2]:
                    attack_codes_nums+=1
                
                if succ_times:
                    self.logger.info(f"Curr succ rate ={succ_times}/{(attack_codes_nums)}=>{succ_times/(attack_codes_nums):3f}, Avg time cost ={total_time:1f}/{succ_times}=>{total_time/succ_times:1f},Avg invo={total_invotions}/{succ_times}=>{total_invotions/succ_times:1f},Avg gen_advs={total_advs_nums}/{succ_times}=>{total_advs_nums/succ_times:1f}, Adv changes nums={total_changes_nums}/{succ_times}=>{total_changes_nums/succ_times:1f} \n")
                else:
                    self.logger.info(f"Curr succ rate = 0, Avg time cost = NaN sec, Avg invo = NaN\n")
            except:
                pass
                # file = self.args.error_log_path
                # with open(file, 'a', encoding='utf-8') as fileobj:
                #     fileobj.write(str(i+last_code_idx+1)+'\n')
            if attack_codes_nums!=0:
                tbar.set_description(f"Curr succ rate {succ_times/(attack_codes_nums):3f}")  
        self.logger.info(f"[Task Done] Time Cost: {time.time()-st_time:1f} sec Succ Rate: {succ_times/attack_codes_nums:.3f}\n")       
        if is_enhance_data:
            self.logger.info("Enhance data Number: %d (Out of %d False Predicted data)" % (fail_pred_num, len(self.enhance_data)))
            if args.sample_nums <=0:
                save_path=os.path.join(self.config['adv_data_path'],self.args.attack_model,self.args.attack_data+"_"+self.args.attack_way+".pkl")
            else:
                save_path=os.path.join(self.config['adv_data_path'],self.args.attack_model,self.args.attack_data+"_"+self.args.attack_way+"_"+str(args.sample_nums)+".pkl")
            if len(self.enhance_data)>0:
                save_pkl(self.enhance_data,save_path)
                self.logger.warning(f"enhanced data is saved in {save_path}")
            else:
                self.logger.warning(f"Failed to enhance data.0/{args.sample_nums}")
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_model',type=str,default='lstm',help="support lstm,gru")
    parser.add_argument('--attack_way',type=str,default='insert',help="support attack ways [grad_rename,random_rename] ")
    parser.add_argument('--attack_data',type=str,default='test',help="Attack dataset [train/val/test]")
    parser.add_argument('--error_log_path',type=str,default='../save/attack/record_problem.log')
    parser.add_argument('--sample_nums',type=int,default=-1,help="Attack only sampled data")
    parser.add_argument('--enhance_type',type=str,default='new_struct',help='Attacks models enhanced with STRUCK')
    parser.add_argument('--enhance_size',type=int,default=-1,help="Attack a model trained with enhanced data")
    parser.add_argument('--graph_type',type=str,default='ggnn_bi',help='gcn or ggnn_bin')
    args=parser.parse_args()
    attack_model_name=args.attack_model
    attack_way=args.attack_way
    attacl_data=args.attack_data
    if attack_model_name in support_model.keys() and attack_way in attack_map.keys() :
        
        set_seed()
        mycfg=myConfig()
        config=mycfg.config
        logger=logging.getLogger(__name__)
        model_config=config[attack_model_name]
       
        attack_belong=attack_map[args.attack_way]# node attack:mhm random_rename,struct attack:insert
        attack_config=config[attack_belong]
        if attacl_data=='train':
            attack_config['is_enhance_data']=True
       
        logger=logging.getLogger(__name__)
       
        if args.attack_model=='ggnn_simple' and args.graph_type is not None and model_config['graph_type']!=args.graph_type:
            model_config['graph_type']=args.graph_type
        if args.attack_model=='ggnn_simple' and not model_config['graph_type'].startswith('ggnn'):
            attack_model=model_config['graph_type']
            model_config['model_path']=model_config['model_path'].replace('ggnn_s',model_config['graph_type'])
            model_config['load_path']=model_config['load_path'].replace('ggnn_s',model_config['graph_type'])
        else:
            attack_model=args.attack_model
        from datetime import date
        today=date.today().day
        if args.sample_nums >0:
            if args.enhance_size>0:
                log_path=os.path.join(attack_config['log_path'],attack_model,args.attack_data+"_"+args.attack_way+"_"+str(attack_config['n_iter'])+"_"+str(attack_config['n_candidate'])+"_day"+str(today)+f"_{args.enhance_type}{args.enhance_size}"+"_sample"+str(args.sample_nums)+".log")
            else:
                log_path=os.path.join(attack_config['log_path'],attack_model,args.attack_data+"_"+args.attack_way+"_"+str(attack_config['n_iter'])+"_"+str(attack_config['n_candidate'])+"_day"+str(today)+"_sample"+str(args.sample_nums)+".log")
        else:
            if args.enhance_size>0:
                log_path=os.path.join(attack_config['log_path'],args.attack_model,args.attack_data+"_"+args.attack_way+"_"+str(attack_config['n_iter'])+"_"+str(attack_config['n_candidate'])+"_day"+str(today)+f"_{args.enhance_type}{args.enhance_size}.log")
            else:
                log_path=os.path.join(attack_config['log_path'],attack_model,args.attack_data+"_"+args.attack_way+"_"+str(attack_config['n_iter'])+"_"+str(attack_config['n_candidate'])+"_day"+str(today)+".log")
            
        args.error_log_path=os.path.join(attack_config['log_path'],attack_model+"_"+args.attack_data+"_"+args.attack_way+"_day"+str(today)+".log")
       
        attack_config['log_path']=log_path
        log_write_mode=attack_config['log_write_mode']
        set_attack_logger(logger,log_path=attack_config['log_path'],write_mode=log_write_mode)
       
        vocab=vocab_map[attack_model_name]() 
      
        n_gpu,device=set_device(config)
        device=device
        model_config['device']=device
        model_config['n_gpu']=n_gpu
        model_config['vocab']=vocab
        
        
        if attack_model_name =='astnn':
            if model_config['using_word2vec_embedding']:
                model_config['word2vec_weight']=vocab.embeddings
            else: 
                model_config['word2vec_weight']=None
            model_config['vocab_size']=vocab.vocab_size
            model=support_model[attack_model_name](**model_config)
        else:
            model=support_model[attack_model_name](**model_config)
        if args.enhance_size>0:
            if args.sample_nums>0:
                file_name=f'best_{args.enhance_type}_sample{args.sample_nums}_{args.enhance_size}.pt'
            else:
                file_name=f'best_{args.enhance_type}_{args.enhance_size}.pt'
            parameter_path=os.path.join(model_config['model_path'],file_name)
            model_config['load_path']=parameter_path
            
        logger.info(f"load model parameter from {model_config['load_path']}")
        model_parameter=torch.load(model_config['load_path'])
        if n_gpu in [1,2]:# GPU
            if list(model_parameter.keys())[0].startswith('module'):
                model = torch.nn.DataParallel(model,device_ids=[0, 1])
                model.load_state_dict(model_parameter)
            else:
                model.load_state_dict(model_parameter)
        else: # cpu
            model.load_state_dict(torch.load(
                model_parameter, map_location='cpu'))  
        if n_gpu > 1 and not isinstance(model,torch.nn.DataParallel):
            model = torch.nn.DataParallel(model,device_ids=[0, 1])
         
        model.to(device)
        
        if isinstance(model,torch.nn.DataParallel):
            model=model.module
        model.eval()
        
        loss_func=lossFunc_map[model_config['loss_func']]()
        
        target_dataset=support_dataset[attack_model_name](vocab,attacl_data,attack_model_name,config=config)
        # file = args.error_log_path
        # with open(file, 'a', encoding='utf-8') as fileobj:
        #     fileobj.write(f"{attack_model_name} Problem code record\n")
        if attack_way in ["insert","random_insert"]:
            
            structAttack=StructAttack(target_dataset,model,loss_func,vocab,device,logger,args,**attack_config)
            
            structAttack.attack_dataset()
        elif attack_way=='mhm':
            mhmAttack=MHMAttack(target_dataset,model,vocab,device,logger,args,**attack_config)
            mhmAttack.attack_dataset()
        else:
           
            renameAttack=RenameAttack(target_dataset,model,loss_func,vocab,device,logger,args,**attack_config)
            
            renameAttack.attack_dataset()