"""struck"""
import sys
sys.path.append("..")

from dataset.CodeFeature import CodeFeatures, CodeFeatures_tbcnn, CodeFeatures_astnn, CodeFeatures_codebert, CodeFeatures_graphcodebert
from dataset.GraphDataset import GGNNDataset_s, Batch_s
# from torch_geometric.data import Batch as Batch
import copy
from random import sample
import time
from config.support import *
from utils.pattern import *
from utils.tools import *
from utils.basic_setting import *
from model.GraphClassify_s import GraphClassifier
from model.GraphCodeBert import GraphCodeBERTClassifier
from model.CodeBert import CodeBERTClassifier
from model.GRU import GRUClassifier
from model.LSTM import LSTMClassifier
import torch
import argparse

from collections import OrderedDict 
sys.path.append("/home/yxzhang/code_atatck/STRUCK/struck")
from get_code_info import *


def predicte_ggnn_list(model, vocab, codes, label, device):
    codefeature_list = [
        CodeFeatures_ggnn(-1, " ".join(code), label, vocab) for code in codes]
    
    input_batch = GGNNDataset_s.get_model_input(
        codefeature_list, vocab)  
    input_batch = GGNNDataset_s.vectorize_input(
        input_batch, training=False, device=device)
    with torch.no_grad():
        if isinstance(model,torch.nn.DataParallel):
            new_probs = model.module.prob(input_batch)
        else:
            new_probs = model.prob(input_batch)
    
    new_preds = torch.argmax(new_probs, dim=1)
    return new_preds, new_probs


def predicte_ggnn(model, vocab, code, label, device):
    # ===
    codefeatures = CodeFeatures_ggnn(-1, code, label, vocab)
    model.to(device)
    input_batch = GGNNDataset_s.get_model_input(
        codefeatures, vocab)  
    input_batch = GGNNDataset_s.vectorize_input(
        input_batch, training=False, device=device)
    label = codefeatures.label
    
    with torch.no_grad():
        if isinstance(model,torch.nn.DataParallel):
            old_probs = model.module.prob(input_batch)
        else:
            old_probs = model.prob(input_batch)
    old_probs = old_probs.squeeze()
    old_pred = torch.argmax(old_probs)
    if label == old_pred:
        return True, old_pred, old_probs[label]
    else:
        return False, old_pred, old_probs[label]


def predicte_lstm_list(model, vocab, codes, label, device):
   
    codefeature_list=[]
    for code in codes:
        codefeature_list.append(CodeFeatures(-1, " ".join(code), label, vocab))
    
    input_idxs, masks, labels = get_input_from_codefeatures_list(
        codefeature_list, device)
    with torch.no_grad():
        if isinstance(model,torch.nn.DataParallel):
            new_probs = model.module.prob(input_idxs, masks)
        else:
            new_probs = model.prob(input_idxs, masks)
    new_preds = torch.argmax(new_probs, dim=1)
    return new_preds, new_probs


def predicte_lstm(model, vocab, code, label, device):
    # ===
    codefeatures = CodeFeatures(-1, code, label, vocab)
    model.to(device)
    
    input_idx, mask, label = get_input_from_codefeatures_list(
        [codefeatures], device)
    label = label[0]  
    
    with torch.no_grad():
        if isinstance(model,torch.nn.DataParallel):
            old_probs = model.module.prob(input_idx, mask).squeeze()
        else:
            old_probs = model.prob(input_idx, mask).squeeze()
    old_pred = torch.argmax(old_probs)
    if label == old_pred:
        return True, old_pred, old_probs[label]

    else:
        return False, old_pred, old_probs[label]



def predicte_codebert_list(model, vocab, codes, label, device):
    codefeatures_list = [
        CodeFeatures_codebert(-1, " ".join(code), label, vocab) for code in codes]
    
    input, _ = CodeBertDataset.get_model_input(codefeatures_list)
    input = input.to(device)
    with torch.no_grad():
        if isinstance(model,torch.nn.DataParallel):
            new_probs = model.module.prob(input)
        else:
            new_probs = model.prob(input)
    
   
    new_preds = torch.argmax(new_probs, dim=1)
    return new_preds, new_probs


def predicte_codebert(model, vocab, code, label, device):
    # ===
    codefeatures = CodeFeatures_codebert(-1, code, label, vocab)
    model.to(device)
    (input_ids, labels) = CodeBertDataset.get_model_input(
        codefeatures)  
    input_ids = input_ids.to(device)
    label = labels[0]  
    
    with torch.no_grad():
        if isinstance(model,torch.nn.DataParallel):
            old_probs = model.module.prob(input_ids).squeeze()
        else:
            old_probs = model.prob(input_ids).squeeze()
    
    old_pred = torch.argmax(old_probs)
    if label == old_pred.item():
        return True, old_pred, old_probs[label]

    else:
        return False, old_pred, old_probs[label]


def predicte_graphcodebert_list(model, vocab, codes, label, device):
    codefeatures_list = [
        CodeFeatures_graphcodebert(-1, " ".join(code), label, vocab) for code in codes]
    
    (inputs_ids, position_idx, attn_mask,
     labels) = GraphCodeBertDataset.get_model_input(codefeatures_list, device)
    with torch.no_grad():
        if isinstance(model,torch.nn.DataParallel):
            new_probs,_ = model.module.prob(inputs_ids, position_idx, attn_mask, labels)
        else:
            new_probs, _ = model.prob(inputs_ids, position_idx, attn_mask, labels)
    
    
    new_preds = torch.argmax(new_probs, dim=1)
    return new_preds, new_probs


def predicte_graphcodebert(model, vocab, code, label, device):
   
    codefeatures = CodeFeatures_graphcodebert(-1, code, label, vocab)
    model.to(device)
    (inputs_ids, position_idx, attn_mask,
     labels) = GraphCodeBertDataset.get_model_input(codefeatures, device)
    label = labels[0]  
    
    with torch.no_grad():
        if isinstance(model,torch.nn.DataParallel):
            old_probs,_ = model.module.prob(inputs_ids, position_idx, attn_mask, labels)
        else:
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
way_map={
    1:'moveNode',
    2:'modifyDatablock',
    3:'modifyFuncblock',
    4:'ReconsBlock',
    5:'redundantNode',
    6:'copyNode',
    7:'changeAndRecoverNode',
    8:'addOldDecl',
    9:'addNewDecl',
    10:'addSubGraph'
}
stage1=['moveNode','modifyDatablock','modifyFuncblock','ReconsBlock']
stage2=['redundantNode','copyNode','changeAndRecoverNode','addOldDecl','addNewDecl','addSubGraph']

class NewStruct_attack(object):
    """Attack the test set using struck"""

    def __init__(self, dataset=read_pkl("../data/test/test_raw.pkl"), model=None, vocab=None, device: torch.device = None, logger=None, args=None, **config) -> None:
        
        self.dataset = dataset
        self.attack_model_name = args.attack_model
        self.model = model.to(device)
        if isinstance(self.model,torch.nn.DataParallel):
            self.model=self.model.module
        self.vocab = vocab
        self.logger = logger
        self.device = device
        self.args = args
        self.config = config
        try:
            logger.warning(
                f"STRUCT ATTACK:{args.attack_way} attack for Model:{args.attack_model} DATASET:{args.attack_data} information is saved in {config['log_path']}")
            myConfig.show_config( self.config, name='DesignStructAttack', logger=self.logger)  
        except:
            myConfig.show_config(self.config, name='DesignStructAttack', logger=self.logger)  
            pass
        

    def attack_single_code(self, code, label,is_log=True): 
      
        is_right, old_pred, old_prob = predicti_map[self.attack_model_name](
            self.model, self.vocab, code, label, self.device)
        save_prob = old_prob  
        if not is_right:  
            if is_log:
                self.logger.info("SUCC! Original mistake.")
            return True, code, label, 0,None,None,None
        
        m = modifyStruct(code, config=self.config)
        
        n_candidate = self.config['n_candidate']
        n_iter = self.config['n_iter']  
        attack_times = 0  
        invotions=0
        gen_adv_nums=0
        while attack_times < n_iter:
            attack_times += 1
            
            if 2 in self.config['stage_choose'] and (attack_times==self.config['max_one_iter'] or m.attack_without_attackDict==[]):
                m.attack_stage=2
            if is_log:
                self.logger.info(f'iter:{attack_times}\n')
            if m.attack_stage==1:
                candidate_code_tokens, cur_attack_way, candidate_names, candidate_change_nums = m.modify_code(n_candidate)
            elif m.attack_stage==2:
                candidate_attackDict,cur_attack_way,candidate_names,candidate_change_nums = m.modify_code(n_candidate)
                
                candidate_code_tokens=[GenCodeFromAttackDict(m.codeinfo.code_tokens, _attackDict) for _attackDict in candidate_attackDict]
            
            if len(candidate_code_tokens) > 0:
                invotions+=1
                gen_adv_nums+=len(candidate_code_tokens)
                new_preds, new_probs = predicti_map[self.attack_model_name+'_list'](
                    self.model, self.vocab, candidate_code_tokens, label, self.device)
                for i in range(len(candidate_code_tokens)):
                    
                    if new_preds[i] != old_pred:
                        
                        m.update_codeinfo(candidate_code_tokens[i],cur_attack_way,candidate_names[i],candidate_change_nums[i])
                        if is_log:
                            self.logger.info(
                            f"SUCC!\t {cur_attack_way} => {candidate_names[i]}\t\t invotions=>{invotions}\t\t gen_adv_nums=>{gen_adv_nums}\t\t attack_changes=>{candidate_change_nums[i]}\t\t{label}:{old_prob.item():4f} => {new_preds[i].item()}: {new_probs[i][new_preds[i]].item():4f}\t\t")
                        return True, candidate_code_tokens[i], new_preds[i], 1,invotions,gen_adv_nums,candidate_change_nums[i]
               
                min_correct_prob_idx = torch.argmin(
                    new_probs[:, label])  
                if new_probs[min_correct_prob_idx][label] < save_prob:  
                    if is_log:
                        self.logger.info(
                        f"acc\t {cur_attack_way} => {candidate_names[min_correct_prob_idx]}\t\t{label}({save_prob.item():4f}) => {label}({new_probs[min_correct_prob_idx][label].item():4f})")
                    
                    if m.attack_stage==1:
                        m.update_codeinfo(candidate_code_tokens[min_correct_prob_idx], cur_attack_way,
                                      candidate_names[min_correct_prob_idx], candidate_change_nums[min_correct_prob_idx])
                    elif m.attack_stage==2:
                        m.update_attacDict(candidate_attackDict[min_correct_prob_idx],cur_attack_way,candidate_names[min_correct_prob_idx])
                    
                    save_prob = new_probs[min_correct_prob_idx][label]
                elif random.random()<self.config['mutation_probability']:
                    if is_log:
                        self.logger.info(
                        f"mutation \t {cur_attack_way} => {candidate_names[min_correct_prob_idx]}\t\t{label}({save_prob.item():4f}) => {label}({new_probs[min_correct_prob_idx][label].item():4f})")
                    
                    if m.attack_stage==1:
                        m.update_codeinfo(candidate_code_tokens[min_correct_prob_idx], cur_attack_way,
                                      candidate_names[min_correct_prob_idx], candidate_change_nums[min_correct_prob_idx])
                    elif m.attack_stage==2:
                        
                        m.update_attacDict(candidate_attackDict[min_correct_prob_idx],cur_attack_way,candidate_names[min_correct_prob_idx])
                    save_prob = new_probs[min_correct_prob_idx][label]
                else:
                   
                    if is_log:
                        self.logger.info(
                        f"rej\t {cur_attack_way} attack {set(candidate_names)}")
                    
            else:
                if cur_attack_way is None:
                    if is_log:
                        self.logger.info("There can be no more attacks in stage1 and stage2")
                    break
                if is_log:
                    self.logger.info(
                            f"rej\t {cur_attack_way} attack {set(candidate_names)}")
        if is_log:         
            self.logger.info("FAIL!")  
        return False, None, label, 2,None,None,None
    
   

    def attack_dataset(self):
        """对传入的数据集进行攻击"""
        data_len = len(self.dataset['code'])
        is_enhance_data = self.config['is_enhance_data']  
        succ_times = self.config["last_succ_times"]  
        total_time = self.config["last_total_time"]  
        total_invotions=self.config['last_total_invotions']
        total_advs_nums=self.config['last_advs_nums']
        total_changes_nums=self.config['last_total_changes_nums']
        attack_codes_nums=self.config['last_attack_codes_nums']
        last_code_idx = self.config['last_attack_code_idx']
        st_time = time.time()  
        code_list = list(range(data_len))
       
        if args.sample_nums >0 and args.attack_data  =='train':
            target_code_ids=random.sample(code_list[last_code_idx:],args.sample_nums) 
        else:
            target_code_ids=code_list[last_code_idx:]
        tbar = tqdm(target_code_ids, file=sys.stdout,mininterval=1000)  # 接着开始 
        fail_pred_num=0
       
        enhance_data=[]
        for i, code_idx in enumerate(tbar):
      
            try:
                id = self.dataset['id'][code_idx]
                code = self.dataset['code'][code_idx]
                label = self.dataset['label'][code_idx]
               
                self.logger.info(
                    f"\t{i+last_code_idx+1}/{data_len}\t ID = {str(label)}_{str(id)}\tY = {label}")
                start_time = time.time() 
               
                is_succ, adv_code, adv_label, result_type,invotions,gen_adv_nums,changes_nums= self.attack_single_code(code, label)
                if is_succ:
                    fail_pred_num+=1

                if result_type==1:
                    succ_times+=1
                    total_time+=time.time()-start_time
                    total_invotions+=invotions
                    total_advs_nums+=gen_adv_nums
                    total_changes_nums+=changes_nums
                    if is_enhance_data and result_type==1:
                        enhance_data.append(codefeatures_map[self.attack_model_name](code_id=id,code=" ".join(adv_code),label=label,vocab=self.vocab))
                        if len(enhance_data) in [1,2000,2500,5000,10000,20000]:
                            if self.args.attack_model=='ggnn_simple':
                                temp_save_path=os.path.join(self.config['adv_data_path'],model_config['graph_type'],self.args.attack_data+"_"+self.args.attack_way+"_"+str(len(enhance_data))+".pkl")
                            else:
                                temp_save_path=os.path.join(self.config['adv_data_path'],self.args.attack_model,self.args.attack_data+"_"+self.args.attack_way+"_"+str(len(enhance_data))+".pkl")
                            save_pkl(enhance_data, temp_save_path)
                            self.logger.warning(f"enhanced {len(enhance_data)} data is saved in {temp_save_path}")
                        
                if result_type in [1,2]:
                    attack_codes_nums+=1
                if result_type==-1:
                    print(f'{code_idx}代码本身解析报错')
                    self.logger.info(f'{code_idx}代码本身解析报错')
                
                if succ_times:
                    self.logger.info(f"Curr succ rate ={succ_times}/{(attack_codes_nums)}=>{succ_times/(attack_codes_nums):3f}, Avg time cost ={total_time:1f}/{succ_times}=>{total_time/succ_times:1f},Avg invo={total_invotions}/{succ_times}=>{total_invotions/succ_times:1f},Avg gen_advs={total_advs_nums}/{succ_times}=>{total_advs_nums/succ_times:1f}, Adv changes nums={total_changes_nums}/{succ_times}=>{total_changes_nums/succ_times:1f} \n")
                else:
                    self.logger.info(f"Curr succ rate = 0, Avg time cost = NaN sec, Avg invo = NaN\n")
                               
            except:
               
                is_right, old_pred, old_prob = predicti_map[self.attack_model_name](self.model, self.vocab, code, label, self.device)
                if is_right:
                    attack_codes_nums+=1
                file = self.args.error_log_path
                with open(file, 'a', encoding='utf-8') as fileobj:
                    fileobj.write(str(code_idx)+'\n')
                
            if attack_codes_nums!=0:      
                tbar.set_description(f"Curr succ rate {succ_times/(attack_codes_nums):3f}") 
        if attack_codes_nums!=0:
            self.logger.info(
                f"[Task Done] Time Cost: {time.time()-st_time:1f} sec Succ Rate: {succ_times/attack_codes_nums:.3f}\n")
        else:
            self.logger.info(
                f"[Task Done] Time Cost: {time.time()-st_time:1f} sec Succ Rate: None\n")
        if is_enhance_data:
            if self.args.attack_model=='ggnn_simple':
                model_file=model_config['graph_type']
            else:
                model_file=self.args.attack_model
            if args.sample_nums <=0:
                save_path=os.path.join(self.config['adv_data_path'], model_file,self.args.attack_data+"_"+self.args.attack_way+".pkl")
            else:
                save_path=os.path.join(self.config['adv_data_path'], model_file,self.args.attack_data+"_"+self.args.attack_way+"_"+str(args.sample_nums)+".pkl")
            
            self.logger.info("Enhance data Number: %d (Out of %d False Predicted data)" % (len(enhance_data),fail_pred_num))
            if len(enhance_data)>0:
                save_pkl(enhance_data, save_path)
                self.logger.warning(f"enhanced data is saved in {save_path}")
            else:
                self.logger.warning(f"Failed to enhance data.0/{args.sample_nums}")
        
    

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_model', type=str,
                        default='lstm', help="support lstm,gru")
    parser.add_argument('--attack_way', type=str, default='new_struct',
                        help="support attack ways [grad_rename,random_rename] ")
    parser.add_argument('--attack_data', type=str,default='test', help="攻击的数据集")
    parser.add_argument('--log_path',type=str,default=None,help='Specially specified debug log path')
    parser.add_argument('--error_log_path',type=str,default='../save/attack/record_problem.log')
    parser.add_argument('--stage_choose',type=int,default=0,help='Specify the stage by running the command line, easy to run, 0 is 1 and 2,1 is 1, 2 is 2, -1 is specified in the file')
    parser.add_argument('--sample_nums',type=int,default=-1,help="Only attack the sampled data, which means no sampling when it is less than 0.")
    parser.add_argument('--enhance_size',type=int,default=-1,help="Attack models trained with enhanced data")
    parser.add_argument('--attack_choose',type=str,default=None,help="Select the specified attack mode for ablation experiments")
    parser.add_argument('--graph_type',type=str,default='ggnn_bi',help='Select gcn or ggnn_bi')
    parser.add_argument('--mutation_probability',type=float,default=0,help='Mutation probability')
    # attack_choose
    """
    0: all，
    1'moveNode', '2 modifyDatablock',' 3 modifyFuncblock', 4 ReconsBlock'
    5'redundantNode', 6'copyNode', 7 'changeAndRecoverNode', 8 'addOldDecl',9 'addNewDecl', 10  'addSubGraph'
    """
    args = parser.parse_args()
    attack_model_name = args.attack_model
    attack_way = args.attack_way
    attacl_data = args.attack_data

    if attack_model_name in support_model.keys() and attack_way in attack_map.keys():
        
        set_seed()
        mycfg = myConfig()
        config = mycfg.config
        logger = logging.getLogger(__name__)
        model_config = config[attack_model_name]
        attack_config = config[attack_map[attack_way]]
        attack_config['mutation_probability']=args.mutation_probability
        if attacl_data=='train':
            attack_config['is_enhance_data']=True
        if args.attack_choose is not None:
            if isinstance(args.attack_choose,str):  
                attack_config['attack_choose']=args.attack_choose.split(',')
            elif isinstance(args.attack_choose,list):
                 attack_config['attack_choose']=args.attack_choose
        
        if attack_config['attack_choose']not in [[0],'0',None]:
            attack_config['attack_choose']=[ int(way ) for way in attack_config['attack_choose'] if way not in ['[',']']]
            print(attack_config['attack_choose'])
            
            stage_choose=[]
            for way in attack_config['attack_choose']:
                print(way_map[int(way)])
                if way_map[int(way)] in stage1:
                    stage_choose.append(1)
                elif way_map[int(way)] in stage2:
                    stage_choose.append(2)
            attack_config['stage_choose']=list(set(stage_choose))
        
        else:
            if args.stage_choose !=-1:
                if args.stage_choose==0:
                    attack_config['stage_choose']=[1,2]
                elif args.stage_choose==1:
                    attack_config['stage_choose']=[1]
                elif args.stage_choose==2:
                    attack_config['stage_choose']=[2]
        if attacl_data=='train':
            attack_config['is_enhance_data']=True
        if args.log_path is None:
            import datetime
            day=datetime.date.today().day
            if attack_config['attack_choose']!=[0]:
                stages='way'
                for way in attack_config['attack_choose']:
                    stages+=str(way)
            else:
                stages='stage'
                for choose in attack_config['stage_choose']:
                    stages+=str(choose)
            
            if args.attack_model=='ggnn_simple' and args.graph_type is not None and model_config['graph_type']!=args.graph_type:
                model_config['graph_type']=args.graph_type
            if args.attack_model=='ggnn_simple' and not model_config['graph_type'].startswith('ggnn'):
                attack_model=model_config['graph_type']
                model_config['model_path']=model_config['model_path'].replace('ggnn_s',model_config['graph_type'])
                model_config['load_path']=model_config['load_path'].replace('ggnn_s',model_config['graph_type'])
            else:
                attack_model=args.attack_model
            if args.enhance_size>0:
                log_path = os.path.join(attack_config['log_path'], attack_model, args.attack_data+"_"+args.attack_way+"_day"+str(day)+"_"+stages+f"_enhance{args.enhance_size}"+f"_m{attack_config['mutation_probability']}"+".log")
            else:
                log_path = os.path.join(attack_config['log_path'], attack_model, args.attack_data+"_"+args.attack_way+"_day"+str(day)+"_"+stages+f"_m{attack_config['mutation_probability']}"+".log")
            args.error_log_path=os.path.join(attack_config['log_path'],attack_model+"_"+args.attack_data+"_"+args.attack_way+"_day"+str(day)+"_"+stages+f"_m{attack_config['mutation_probability']}"+".log")
        else:
            log_path=args.log_path
        
        attack_config['log_path'] = log_path
        log_write_mode = attack_config['log_write_mode']
        set_attack_logger(
            logger, log_path=attack_config['log_path'], write_mode=log_write_mode)
        
        n_gpu, device = set_device(config)
        model_config['device'] = device
        vocab = vocab_map[attack_model_name](config=config)
        model_config['vocab'] = vocab
        if attack_model_name == 'astnn':
            if model_config['using_word2vec_embedding']:
                model_config['word2vec_weight'] = vocab.embeddings
            else:
                model_config['word2vec_weight'] = None
            model_config['vocab_size'] = vocab.vocab_size
        
        model = support_model[attack_model_name](**model_config)  
        if args.enhance_size>0:
            if args.sample_nums>0:
                file_name=f'best_{args.attack_way}_sample{args.sample_nums}_{args.enhance_size}.pt'
            else:
                file_name=f'best_{args.attack_way}_{args.enhance_size}.pt'
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
                model_parameter, map_location='cpu'))  # cpu上
        if n_gpu > 1 and not isinstance(model,torch.nn.DataParallel):
            model = torch.nn.DataParallel(model,device_ids=[0, 1])
        

        model.eval()
        data_path='../data/target/target_raw.pkl'
        data_path=data_path.replace('target',attacl_data)
        dataset=read_pkl(data_path)
        attack = NewStruct_attack(dataset=dataset,
            model=model, vocab=vocab, device=device, logger=logger, args=args, **attack_config)
        
        file = args.error_log_path
        # with open(file, 'a', encoding='utf-8') as fileobj:
        #     fileobj.write(f"{attack_model_name} problem code record\n")
        attack.attack_dataset()
        
        
