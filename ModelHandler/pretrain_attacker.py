from inspect import ArgSpec
from pyexpat import model
from re import sub
import sys
sys.path.append("..")
import torch
from dataset.CodeFeature import CodeFeatures,CodeFeatures_tbcnn,CodeFeatures_astnn,CodeFeatures_codebert
from dataset.vocabClass import VocabModel,VocabModel_tbcnn,VocabModel_astnn,VocabModel_codebert
from model.LSTM import LSTMClassifier
from model.GRU import GRUClassifier
from model.TBCNN import TBCNNClassifier
from model.ASTNN import BatchProgramClassifier
from model.CodeBert import CodeBERTClassifier
from utils.basic_setting import *
from utils.tools import *
from utils.pattern import *
import argparse 
from config.support import *
import time
from random import sample
import copy
# from torch_geometric.data import Batch
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
import string
def predicte_codebert_list(model, vocab, codes,label, device):
    codefeature_list = [
        CodeFeatures_codebert(-1, " ".join(code), label, vocab) for code in codes]
    
    input, _ = CodeBertDataset.get_model_input(codefeature_list)
    input = input.to(device)
    new_probs = model.prob(input)
   
    new_preds = torch.argmax(new_probs, dim=1)
    return new_preds, new_probs,codefeature_list


def predicte_codebert(model, vocab, codefeatures,label, device):
   
    if isinstance(codefeatures,list):
        codefeatures = CodeFeatures_codebert(-1, " ".join(codefeatures), label, vocab)
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


def predicte_graphcodebert(model, vocab, codefeatures,label,  device):
  
    if isinstance(codefeatures,list):
        codefeatures = CodeFeatures_graphcodebert(-1, " ".join(codefeatures), label, vocab)
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
    'codebert_list': predicte_codebert_list,
    'graphcodebert_list': predicte_graphcodebert_list,
    

  
    'codebert': predicte_codebert,
    'graphcodebert': predicte_graphcodebert,
    
}
codefeature_map={
    'lstm': CodeFeatures,
    'gru': CodeFeatures,
    'tbcnn': CodeFeatures_tbcnn,
    'astnn': CodeFeatures_astnn,
    'codebert': CodeFeatures_codebert,
    'graphcodebert': CodeFeatures_graphcodebert,
    'ggnn_simple': CodeFeatures_ggnn, 
}

class LocPosAttack(object):
    def  __init__(self,device:torch.device=None,args=None,config=None) -> None:
        
        self.device=device
        self.args=args
        self.config=config
        ## Load CodeBERT (MLM) model
        self.codebert_mlm = RobertaForMaskedLM.from_pretrained(args.pretrain_model)
        self.tokenizer_mlm = RobertaTokenizer.from_pretrained(args.pretrain_model)
        self.codebert_mlm.to(device) 
    def get_target_token_substitutes(self,codefeatures:CodeFeatures_codebert,):      
        
        sub_tokens,subtokens_pos=self.get_code_subtokens(codefeatures.tokens,self.tokenizer_mlm)
        sub_tokens = [self.tokenizer_mlm.cls_token] + sub_tokens[:args.block_size - 2] + [self.tokenizer_mlm.sep_token]
        input_ids_ = torch.tensor([self.tokenizer_mlm.convert_tokens_to_ids(sub_tokens)])
        
        word_predictions = self.codebert_mlm(input_ids_.to(self.device)).logits.squeeze()  # [seq_len(sub),vocab] 每个subtoken对vocab中每个token的预测值 
        # topk word_predications
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.config['top_similar_candidate'], -1)  # [seq_len,k] 
        

        word_predictions = word_predictions[1:len(sub_tokens) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_tokens) + 1, :]
        
        names_positions_dict = self.get_identifier_posistions_from_code(codefeatures.tokens, codefeatures.target_tokens)

        variable_substitue_dict = {}
        with torch.no_grad():
            orig_embeddings = (self.codebert_mlm.roberta(input_ids_.to(self.device)).last_hidden_state)[0]# [1, token_nums, embedding_size]
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for tgt_word in names_positions_dict.keys():
            if tgt_word in forbidden_tokens:
                continue
            is_illeagal=True
            for c in tgt_word:
                if c not in string.ascii_letters and c not in string.digits:
                    is_illeagal=False
                    break
            if not is_illeagal:continue
            tgt_positions = names_positions_dict[tgt_word] # the positions of tgt_word in code
           
            all_substitues = []
            for one_pos in tgt_positions:
                
                if subtokens_pos[one_pos][0] >= word_predictions.size()[0]:
                    continue
                substitutes = word_predictions[subtokens_pos[one_pos][0]:subtokens_pos[one_pos][1]] 
                word_pred_scores = word_pred_scores_all[subtokens_pos[one_pos][0]:subtokens_pos[one_pos][1]]
                orig_word_embed = orig_embeddings[subtokens_pos[one_pos][0]+1:subtokens_pos[one_pos][1]+1]
                similar_substitutes = []
                similar_word_pred_scores = []
                sims = []
                subwords_leng, nums_candis = substitutes.size()
                for i in range(nums_candis):
                    new_ids_ = copy.deepcopy(input_ids_)
                    new_ids_[0][subtokens_pos[one_pos][0]+1:subtokens_pos[one_pos][1]+1] = substitutes[:,i]
                    
                    with torch.no_grad():
                        new_embeddings = self.codebert_mlm.roberta(new_ids_.to(self.device))[0]
                    new_word_embed = new_embeddings[0][subtokens_pos[one_pos][0]+1:subtokens_pos[one_pos][1]+1]
                    sims.append((i, sum(cos(orig_word_embed, new_word_embed))/subwords_leng))
                sims = sorted(sims, key=lambda x: x[1], reverse=True)
               
                for i in range(int(nums_candis/2)):
                    similar_substitutes.append(substitutes[:,sims[i][0]].reshape(subwords_leng, -1))
                    similar_word_pred_scores.append(word_pred_scores[:,sims[i][0]].reshape(subwords_leng, -1))

                similar_substitutes = torch.cat(similar_substitutes, 1)
                similar_word_pred_scores = torch.cat(similar_word_pred_scores, 1)
               
                substitutes = self.get_substitues(similar_substitutes, 
                                            self.tokenizer_mlm, 
                                            self.codebert_mlm, 
                                            device=device,
                                            use_bpe=1, 
                                            substitutes_score=similar_word_pred_scores, 
                                            threshold=self.config['similar_threshold'])
                all_substitues += substitutes
            all_substitues = set(all_substitues)

            for tmp_substitue in all_substitues:
                if tmp_substitue.strip()=='':
                    continue 
                if tmp_substitue.strip() in codefeatures.target_tokens:
                    continue
                if tmp_substitue.strip() in forbidden_tokens:
                    continue
                if vocab.getToken(tmp_substitue.strip())!=tmp_substitue.strip():
                    continue 
                if tmp_substitue[0] in ops:
                    continue
                try:
                    variable_substitue_dict[tgt_word].append(tmp_substitue)
                except:
                    variable_substitue_dict[tgt_word] = [tmp_substitue]
        # 返回这个代码的变量替换字典
        return variable_substitue_dict
    @ classmethod
    def get_substitues(cls,substitutes, tokenizer, mlm_model,use_bpe,device,substitutes_score=None, threshold=3.0):
        
        # substitues L,k
        # from this matrix to recover a word
        words = []
        sub_len, k = substitutes.size()  # sub_len, k

        if sub_len == 0:
            
            return words

        elif sub_len == 1:
            
            for (i, j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(tokenizer._decode([int(i)]))
                
        else:
            
            if use_bpe == 1:
                words = cls.get_bpe_substitues(substitutes, tokenizer, mlm_model,device)
            else:
                return words
        return words
    @staticmethod
    def get_bpe_substitues(substitutes, tokenizer, mlm_model,device):
       
        # substitutes L, k

        substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

        # find all possible candidates 

        all_substitutes = []
        for i in range(substitutes.size(0)):
            if len(all_substitutes) == 0:
                lev_i = substitutes[i]
                all_substitutes = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_sub in all_substitutes[:24]:  
                    for j in substitutes[i]:
                        lev_i.append(all_sub + [int(j)])
                all_substitutes = lev_i
        # all substitutes  list of list of token-id (all candidates)
        c_loss = nn.CrossEntropyLoss(reduction='none')
        word_list = []
        # all_substitutes = all_substitutes[:24]
        all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
        all_substitutes = all_substitutes[:24].to(device)
       
        N, L = all_substitutes.size()
        word_predictions = mlm_model(all_substitutes.long())[0]  # N L vocab-size
        ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.long().view(-1))  # [ N*L ]
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
        _, word_list = torch.sort(ppl)
        word_list = [all_substitutes[i] for i in word_list]
        final_words = []
        for word in word_list:
            tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
            text = tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        return final_words

    @staticmethod
    def get_code_subtokens(code_tokens,tokenizer):
        
        sub_words = []
        keys = []
        index = 0
        for word in code_tokens:
            
            sub = tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return sub_words, keys
    @staticmethod
    def get_identifier_posistions_from_code(words_list: list, variable_names: list) -> dict:
      
        positions = {}
        for name in variable_names:
            for index, token in enumerate(words_list):
                if name == token:
                    try:
                        positions[name].append(index)
                    except:
                        positions[name] = [index]

        return positions
    @staticmethod
    def get_masked_code_by_position(tokens: list, positions: dict):
       
        masked_token_list = []
        replace_token_positions = []
        for variable_name in positions.keys():
            for pos in positions[variable_name]:
                masked_token_list.append(tokens[0:pos] + ['<unk>'] + tokens[pos + 1:])
                replace_token_positions.append(pos)
        
        return masked_token_list, replace_token_positions
    @classmethod
    def get_importance_score_codebert(cls,args, codefeatures:CodeFeatures_codebert, variable_names: list, model,vocab,device):
       
        # label: example[1] tensor(1)
        
        positions = cls.get_identifier_posistions_from_code(codefeatures.tokens, variable_names)
       
        if len(positions) == 0:
           
            return None, None, None

        new_inputs = [codefeatures.input_ids]

       
        masked_token_list, replace_token_positions = cls.get_masked_code_by_position(codefeatures.tokens, positions)
        
        for index, tokens in enumerate(masked_token_list):
             new_inputs.append(vocab.tokenize(' '.join(tokens),cut_and_pad=True,ret_id=True)[0])
             
       
        
        new_inputs=torch.tensor(new_inputs).to(device)
        new_probs=model.prob(new_inputs)
        new_preds=torch.argmax(new_probs,dim=1)
        old_pred=new_preds[0]
        old_prob = new_probs[0][old_pred]
        importance_score = []
        for prob in new_probs[1:]:
            importance_score.append(old_prob - prob[old_pred])
        
        return importance_score, replace_token_positions, positions
    @classmethod
    def get_importance_score_graphcodebert(cls,args, codefeatures:CodeFeatures_codebert, variable_names: list, model,vocab,device):
        '''Compute the importance score of each variable'''
        # label: example[1] tensor(1)
       
        positions = cls.get_identifier_posistions_from_code(codefeatures.tokens, variable_names)
        
        if len(positions) == 0:
            
            return None, None, None

        

        # 2. Masked_tokens,masked_token_list
        masked_token_list, replace_token_positions = cls.get_masked_code_by_position(codefeatures.tokens, positions)
        # replace_token_positions 
        
        # 3.Construct a new input
        input_ids_l,position_idxs_l,attn_mask_l,labels_l=[],[],[],[]
        for index, tokens in enumerate([codefeatures.tokens] + masked_token_list):
            new_code = ' '.join(tokens)
            input_tokens,input_ids,position_idx,dfg_to_code,dfg_to_dfg = vocab.convert_examples_to_features(new_code, vocab.parser,vocab.tokenizer,"c",config['graphcodebert'])
            input_ids_l.append(input_ids)
            position_idxs_l.append(position_idx)
            attn_mask_l.append(GraphCodeBertDataset.generate_atten_mask_from_input(input_ids,position_idx,dfg_to_code,dfg_to_dfg))
            labels_l.append(codefeatures.label)
              
        input_ids_l=torch.tensor(input_ids_l).to(device)
        position_idxs_l=torch.tensor(position_idxs_l).to(device)
        attn_mask_l=torch.tensor(attn_mask_l).to(device)
        labels_l=torch.tensor(labels_l).to(device)
        with torch.no_grad():
            if isinstance(model,torch.nn.DataParallel):
                new_probs,_ = model.module.prob(input_ids_l, position_idxs_l, attn_mask_l, labels_l)
            else:
                new_probs, _ = model.prob(input_ids_l, position_idxs_l, attn_mask_l, labels_l)
             
        
        
        new_preds=torch.argmax(new_probs,dim=1)
        old_pred=new_preds[0] 
        old_prob = new_probs[0][old_pred]
        importance_score = []
        for prob in new_probs[1:]:
            importance_score.append(old_prob - prob[old_pred])
        
        return importance_score, replace_token_positions, positions
    @staticmethod
    def get_new_codefeature(codefeatures:CodeFeatures_codebert,target_token,rename_token,vocab,target_model_name='codebert'):
        tokens=codefeatures.tokens
        for i,token in enumerate(tokens):
            if token==target_token:
                tokens[i]=rename_token
        code=tokens2code(tokens)
        
        return codefeature_map[target_model_name](codefeatures.id,code,codefeatures.label,vocab,rename=True)
    @staticmethod
    def get_new_code_tokens(tokens,target_token,rename_token):
        for i,token in enumerate(tokens):
            if token==target_token:
                tokens[i]=rename_token
        return tokens
    @staticmethod
    def get_new_code(tokens,target_token,rename_token):
        for i,token in enumerate(tokens):
            if token==target_token:
                tokens[i]=rename_token
        code=tokens2code(tokens)
        return code
    @staticmethod
    def select_parents(population): 
        if len(population)==1:
            return population[0],0,population[0],0
        length = range(len(population))
        [index_1,index_2] = random.sample(length,2)
        chromesome_1 = population[index_1]
        chromesome_2 = population[index_2]
        return chromesome_1, index_1, chromesome_2, index_2
    @staticmethod
    def mutate(chromesome, variable_substitue_dict):
        tgt_index = random.choice(range(len(chromesome)))
        tgt_word = list(chromesome.keys())[tgt_index]
        fail_times=10
        chromesome[tgt_word]=tgt_word
        while fail_times:
            is_illegal=True
            candi=random.choice(variable_substitue_dict[tgt_word])
            for c in candi:
                if c not in string.ascii_letters and c not in string.digits and c not in ['_',' ']:
                    is_illegal=False
                    break
            if is_illegal:
                chromesome[tgt_word] = candi
                break
            fail_times-=1
                    

        return chromesome
    @staticmethod
    def crossover(csome_1, csome_2, cut_off_position=None):
        
        if cut_off_position is None:
            cut_off_position = random.choice(range(1,len(csome_1))) 
            

        child_1 = {}
        child_2 = {}
        for index, variable_name in enumerate(csome_1.keys()):
            if index < cut_off_position: 
                child_2[variable_name] = csome_1[variable_name]
                child_1[variable_name] = csome_2[variable_name]
            else:
                child_1[variable_name] = csome_1[variable_name]
                child_2[variable_name] = csome_2[variable_name]
        return child_1, child_2
    @staticmethod
    def check_valid_chromesome(chromesome:dict,valid_tokens=None):
        
        if valid_tokens is None:
            valid_tokens=list(chromesome.keys())
        used_token=[]
        for key in chromesome.keys():
            if chromesome[key]!=key and chromesome[key] in valid_tokens:
                return False
            if chromesome[key] in used_token :
                return False
            used_token.append(chromesome[key])
        return True
    @staticmethod
    def compute_fitness(chromesome, model, vocab, orig_prob, orig_label ,tokens,device,attack_model_name):
       
        temp_code = LocPosAttack.map_chromesome(chromesome, tokens)
        temp_input= [vocab.tokenize(temp_code,cut_and_pad=True,ret_id=True)[0]]
        temp_input=torch.tensor(temp_input).to(device)
        new_prob=model.prob( temp_input)[0]
        new_pred=torch.argmax(new_prob)
        
        fitness_value = orig_prob - new_prob[orig_label]
        return fitness_value, new_pred
    @staticmethod
    def compute_fitness_tokens(chromesome, model, vocab, orig_prob, orig_label ,tokens,device,attack_model_name):
        
        temp_code = LocPosAttack.map_chromesome_tokens(chromesome, tokens)
        is_right, pred, prob = predicti_map[attack_model_name](model, vocab, temp_code,orig_label,device)
        
        fitness_value = orig_prob - prob
        return fitness_value, pred
    @staticmethod
    def map_chromesome(chromesome: dict, tokens: list):
        for target_word in chromesome.keys():
            if target_word!=chromesome[target_word]:
               tokens=LocPosAttack.get_new_code_tokens(tokens,target_word,chromesome[target_word]) 
        temp_replace = " ".join(tokens)
        return temp_replace
    @staticmethod
    def map_chromesome_tokens(chromesome: dict, tokens: list):
        for target_word in chromesome.keys():
            if target_word!=chromesome[target_word]:
               tokens=LocPosAttack.get_new_code_tokens(tokens,target_word,chromesome[target_word]) 
        
        return tokens
    @staticmethod
    def show_population(logger,population):
        
        population_str=f"the population length is {len(population)},they are:"
        for idx,p in enumerate(population):
            ori=True
            population_str+=f"\n\t {idx+1}: \t"
            for key in p.keys():
                if key!=p[key]:
                    population_str+=f"{key}=>{p[key]}\t"
                    ori=False
            if ori:
                population_str+='ori'
        logger.info(population_str)
class pretrainAttack(object):
    def __init__(self,dataset:CodeBertDataset,model:CodeBERTClassifier, vocab:VocabModel_codebert, device=None,logger=None, args=None,**config) -> None:
        self.dataset=dataset
        self.args = args
        self.attack_model_name = args.attack_model
        self.config=config
        self.model = model.to(device)
        self.vocab = vocab
        self.device=device
        self.use_bpe = config['use_bpe'] 
        self.logger=logger
        self.enhance_data=[]
       
        self.locpos=LocPosAttack(device=self.device,args=self.args,config=self.config) 
        logger.warning(
            f"STRUCT ATTACK:{args.attack_way} attack for Model:{args.attack_model} DATASET:{args.attack_data} information is saved in {config['log_path']}")
        myConfig.show_config(
            self.config, name='Natural attack', logger=self.logger)  
    def greedy_attack(self, codefeatures:CodeFeatures_codebert, substitutes):
        
        res=single_code_node_attack_result(ori_code=codefeatures)
        
        is_right, old_pred, old_prob = predicti_map[self.attack_model_name](self.model, self.vocab, codefeatures,codefeatures.label,self.device)
        
        label=codefeatures.label
       
        
        
        
        
        variable_names = list(substitutes.keys())

        if old_pred != label:
           
            res.result_type=-1
            return  res
            
        if len(variable_names) == 0:
            
            res.result_type=0
            return res,None,None

       
        invotions=0
        gen_adv_nums=0
        if self.attack_model_name =='codebert':
            importance_score, replace_token_positions, names_positions_dict =LocPosAttack.get_importance_score_codebert(self.args, codefeatures,variable_names=variable_names,model=self.model,vocab=self.vocab,device=self.device)
        elif self.attack_model_name =='graphcodebert':
            importance_score, replace_token_positions, names_positions_dict =LocPosAttack.get_importance_score_graphcodebert(self.args, codefeatures,variable_names=variable_names,model=self.model,vocab=self.vocab,device=self.device)
        
        
        if importance_score is None:
            res.result_type=3
            return res
        else:
            invotions+=1
            gen_adv_nums+=len(importance_score)


        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                
                total_score += importance_score[token_pos_to_score_pos[token_pos]]
            
            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
       

        final_codefeature = copy.deepcopy(codefeatures)
        nb_changed_var = 0 
        nb_changed_pos = 0 
        replaced_words = {}
        
        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]
           
            all_substitues = substitutes[tgt_word]

            most_gap = 0.0
            most_gap_idx=0
            candidate = None
           
            candidate_code_tokens=[]
            substitute_list = []
            
            for substitute in all_substitues:
                if substitute in final_codefeature.target_tokens:
                    continue  
                is_illegal=True
                for c in substitute:
                    if c not in string.ascii_letters and c not in string.digits and c not in ['_',' ']:
                        is_illegal=False
                        break
                if not is_illegal:continue
                substitute_list.append(substitute)
                
                
                temp_code = LocPosAttack.get_new_code_tokens(copy.deepcopy(final_codefeature.tokens), tgt_word, substitute) 
                candidate_code_tokens.append(temp_code)                              
               
            if len(candidate_code_tokens) == 0:
                
                continue
            
            # replace_inputs=torch.tensor(replace_inputs).to(self.device)
            invotions+=1
            gen_adv_nums+=len(candidate_code_tokens)
            new_preds, new_probs,adv_codes = predicti_map[self.attack_model_name+'_list'](self.model, self.vocab, candidate_code_tokens, label, self.device)
            # new_probs=self.model.prob(replace_inputs)
            # new_preds=torch.argmax(new_probs,dim=1)
            assert(len(new_probs) == len(substitute_list))


            for index, temp_prob in enumerate(new_probs):
                temp_label =new_preds[index]
                if temp_label != label:
                    
                    
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate
                    adv_code=adv_codes[index]
                    # adv_code = LocPosAttack.get_new_codefeature(copy.deepcopy(final_codefeature), tgt_word, candidate,self.vocab,self.attack_model_name)
                    # self.logger.info(f"SUCC!\t {tgt_word} => {candidate}\t\t {label}:{old_prob[label].item():4f} => {temp_label.item()}: {new_probs[temp_label].item():4f}\n" )
                    
                    res.update(succ=True,result_type=1,adv_code=adv_code,new_pred=temp_label,node_changes=replaced_words,names_to_important_score=names_to_importance_score,nb_changed_var=nb_changed_var,nb_changed_pos=nb_changed_pos)
                    
                    return res,invotions,gen_adv_nums
                else:
                    
                    gap = old_prob - temp_prob[temp_label]
                   
                    if gap > most_gap:  
                        if substitute_list[index].isidentifier():
                            most_gap = gap
                            candidate = substitute_list[index]
                            most_gap_idx=index

            if most_gap > 0:

                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = new_probs[most_gap_idx][label]
                replaced_words[tgt_word] = candidate
                try:
                    final_codefeature = LocPosAttack.get_new_codefeature(copy.deepcopy(final_codefeature), tgt_word, candidate,self.vocab,self.attack_model_name)
                except:
                    print("error!")
                self.logger.info(f"acc\t {tgt_word} => {candidate}\t\t{label}({old_prob.item():4f}) => {label}({current_prob:4f})\n" )
            else:
                replaced_words[tgt_word] = tgt_word
        
        res.result_type=3
        res.node_changes=replaced_words
        res.adv_code=final_codefeature
        return res,invotions,gen_adv_nums
    def ga_attack(self,codefeatures:CodeFeatures_codebert,substitutes:dict,initial_replace=None,cross_probability=0.7):

        is_right, old_pred, old_prob = predicti_map[self.attack_model_name](self.model, self.vocab, codefeatures,codefeatures.label,self.device)
        correct_prob=old_prob
        label=codefeatures.label
        
        
        res=single_code_node_attack_result(ori_code=codefeatures)
        
        variable_names = list(substitutes.keys())

        if not is_right:
          
            res.result_type=-1
            return  res
            
        if len(variable_names) == 0:
          
            res.result_type=0
            return res
      
        names_positions_dict = LocPosAttack.get_identifier_posistions_from_code(codefeatures.tokens, variable_names)
        
        
        fitness_values=[] 
       
        base_chromesome={word: word for word in substitutes.keys()}
        # Population Initialization
        population = [base_chromesome] 
        invotions=0
        gen_adv_nums=0
       
        for tgt_word in substitutes.keys():
          
            if initial_replace is not None:
                if initial_replace.get(tgt_word) is not None and initial_replace.get(tgt_word)!=tgt_word:
                    initial_candidate = initial_replace[tgt_word] 
                else:
                    continue 
            else:
                
                candidate_code_tokens=[]
                substitute_list = []
                
                for a_substitue in substitutes[tgt_word]:
                    if a_substitue in codefeatures.target_tokens:
                        continue  
                    is_illegal=True
                    for c in a_substitue:
                        if c not in string.ascii_letters and c not in string.digits and c not in ['_',' ']:
                            is_illegal=False
                            break
                    if not is_illegal:continue
                    substitute_list.append(a_substitue)
                    
                    temp_code = LocPosAttack.get_new_code_tokens(copy.deepcopy(codefeatures.tokens), tgt_word, a_substitue) 
                    candidate_code_tokens.append(temp_code)
                    
                if len(candidate_code_tokens) == 0:
                    
                    continue
              
                invotions+=1
                gen_adv_nums+=len(candidate_code_tokens)
               
                new_preds, new_probs,new_codefeatures = predicti_map[self.attack_model_name+'_list'](
                    self.model, self.vocab, candidate_code_tokens, label, self.device)
                assert(len(new_probs) == len(substitute_list))
                most_gap = 0.0
                best_candidate=-1
                for idx,prob in enumerate(new_probs):
                    temp_label = new_preds[idx]
                    if temp_label!=label:
                        res.update(result_type=2,adv_code=new_codefeatures[idx])
                        return res
                    gap=correct_prob-prob[temp_label]
                   
                    if gap > most_gap:
                        
                        if substitute_list[idx] not in variable_names:
                            most_gap = gap
                            best_candidate = idx
                if best_candidate == -1:
                    initial_candidate = tgt_word
                   
                    continue 
                else:
                    initial_candidate = substitute_list[best_candidate]
           
           
            temp_chromesome = copy.deepcopy(base_chromesome)
            temp_chromesome[tgt_word] = initial_candidate
            population.append(temp_chromesome)
            temp_fitness, temp_label = LocPosAttack.compute_fitness_tokens(tokens=copy.deepcopy(codefeatures.tokens),chromesome=temp_chromesome,model= self.model, vocab=self.vocab, orig_prob=correct_prob, orig_label=old_pred,device=self.device,attack_model_name=self.attack_model_name)
            fitness_values.append(temp_fitness)
        LocPosAttack.show_population(self.logger,population)
       
        if len(fitness_values)==0:
           
            res.result_type=3
            return res,invotions,gen_adv_nums
        try:
            max_iter=self.config['ga_max_iter']
        except:
            max_iter = max(5 * len(population), 10)
            
        for i in range(max_iter):
            temp_mutants=[]
            for j in range(self.config['eval_batch_size']):
                p=random.random()
                chromesome_1, index_1, chromesome_2, index_2 = LocPosAttack.select_parents(population)
                if p < cross_probability:
                    child_1, child_2 = LocPosAttack.crossover(chromesome_1, chromesome_2)
                    if child_1 == chromesome_1 or child_1 == chromesome_2:
                        child_1 = LocPosAttack.mutate(copy.deepcopy(child_1), substitutes)
                else: 
                    child_1 = LocPosAttack.mutate(copy.deepcopy(chromesome_1), substitutes)
                if LocPosAttack.check_valid_chromesome(child_1):
                   temp_mutants.append(child_1)
            
            new_codes=[]
           
            for mutant in temp_mutants:
                temp_code = LocPosAttack.map_chromesome_tokens(mutant,copy.deepcopy(codefeatures.tokens))
                
                new_codes.append(temp_code)
               
            if len(new_codes) == 0:
                continue
            
            mutate_preds, mutate_probs,murate_codefeatures = predicti_map[self.attack_model_name+'_list'](
                    self.model, self.vocab, new_codes, label, self.device)
            
            invotions+=1
            gen_adv_nums+=len(new_codes)
            mutate_preds=torch.argmax(mutate_probs,dim=1)
            mutate_fitness_values = []
            for index, probs in enumerate(mutate_probs):
                if mutate_preds[index] != old_pred:
                    adv_code = new_codes[index]
                    nb_changed_var = 0 
                    nb_changed_pos = 0 
                    for old_word in temp_mutants[index].keys():
                        if old_word != temp_mutants[index][old_word]:
                            nb_changed_var += 1
                            nb_changed_pos += len(names_positions_dict[old_word])
                    
                    adv_code=murate_codefeatures[index]
                    self.logger.info(f"iter {i} {old_pred}{correct_prob} the min lable correct is {correct_prob-max(fitness_values)}")
                    res.update(result_type=2,adv_code=adv_code,new_pred=mutate_preds[index],node_changes=temp_mutants[index],nb_changed_var=nb_changed_var,nb_changed_pos=nb_changed_pos)
                    return res,invotions,gen_adv_nums
                tmp_fitness = correct_prob - probs[old_pred]
                mutate_fitness_values.append(tmp_fitness)
            
            for index, fitness_value in enumerate(mutate_fitness_values):
                min_value = min(fitness_values)
                if fitness_value > min_value:
                    
                    min_index = fitness_values.index(min_value)
                    population[min_index+1] = temp_mutants[index]
                    
                    fitness_values[min_index] = fitness_value
            
      
        self.logger.info(f"iter {i} the best fitness value is {max(fitness_values)} and the min lable correct is {correct_prob-max(fitness_values)}")
        res.result_type=3
        return res,invotions,gen_adv_nums
        
    def attack_single_code(self,codefeatures:CodeFeatures_codebert,substitutes:dict=None,initial_replace=None):
        codefeatures.prepare_for_rename(self.vocab)
        invotions=0
        gen_adv_nums=0
        if substitutes is None:
            substitutes=self.locpos.get_target_token_substitutes(codefeatures)
        if self.config['Greedy']:
            self.logger.info("using Greedy attack")
            res,invotions_1,gen_adv_nums_1=self.greedy_attack(codefeatures,substitutes)
            if invotions_1 is not None:
                invotions+=invotions_1
                gen_adv_nums+=gen_adv_nums_1
        if res.succ:
            return res,invotions,gen_adv_nums
        if  self.config['GA']:
            if self.config['Greedy']:
                self.config['ga_max_iter']=self.config['n_iter']-invotions
                self.logger.info("Greedy attack fails, using GA attack")
            else:
                self.config['ga_max_iter']=self.config['n_iter']
                self.logger.info("using GA attack")
            cross_probability=self.config['cross_probability']
            if res.node_changes is not None:
                initial_replace=res.node_changes
            res,invotions_2,gen_adv_nums_2=self.ga_attack(codefeatures,substitutes,initial_replace,cross_probability)
            invotions+=invotions_2
            gen_adv_nums+=gen_adv_nums_2
        return res,invotions,gen_adv_nums
    def attack_dataset(self,substitutes=None):
        '''
        Description: 攻击一整个数据集
        '''
        data=self.dataset.data
        data_len=len(data)
        is_enhance_data = self.config['is_enhance_data']  
        succ_times = self.config["last_succ_times"]  
        total_time = self.config["last_total_time"]  
        total_invotions=self.config['last_total_invotions']
        total_advs_nums=self.config['last_advs_nums']
        total_changes_nums=self.config['last_total_changes_nums']
        attack_codes_nums=self.config['last_attack_codes_nums']
        last_code_idx = self.config['last_attack_code_idx']
        st_time = time.time()  
    
        if args.sample_nums >0:
            
            targt_dates=random.sample(data[last_code_idx:],args.sample_nums) 
        else:
            targt_dates=data[last_code_idx:]
        tbar = tqdm(targt_dates, file=sys.stdout)
        fail_pred_num=0
        for i, target_codefeatures in enumerate(tbar):
           
            self.logger.info(f"\t{i+last_code_idx+1}/{data_len}\t ID = {target_codefeatures.label+1}_{target_codefeatures.id}\tY = {target_codefeatures.label}")
            start_time = time.time()
            
            try:
                code_substitutes=substitutes[last_code_idx+i]
            except:
                code_substitutes=None
            try:
                res,invotions,gen_adv_nums=self.attack_single_code(target_codefeatures,code_substitutes)
                res.show_result(self.logger)
                if res.succ:
                    fail_pred_num+=1
                if res.result_type in [1,2]:
                    succ_times+=1
                    total_time+=time.time()-start_time
                    total_invotions+=invotions
                    total_advs_nums+=gen_adv_nums
                    total_changes_nums+=res.nb_changed_pos
                    if is_enhance_data:
                        self.enhance_data.append(res.adv_code)
                if res.result_type!=-1:
                    attack_codes_nums+=1
                
                if succ_times:
                    self.logger.info(f"Curr succ rate ={succ_times}/{(attack_codes_nums)}=>{succ_times/(attack_codes_nums):3f}, Avg time cost ={total_time:1f}/{succ_times}=>{total_time/succ_times:1f},Avg invo={total_invotions}/{succ_times}=>{total_invotions/succ_times:1f},Avg gen_advs={total_advs_nums}/{succ_times}=>{total_advs_nums/succ_times:1f}, Adv changes nums={total_changes_nums}/{succ_times}=>{total_changes_nums/succ_times:1f} \n")
                else:
                    self.logger.info(f"Curr succ rate = 0, Avg time cost = NaN sec, Avg invo = NaN\n")
            except:
                file = self.args.error_log_path
                with open(file, 'a', encoding='utf-8') as fileobj:
                    fileobj.write(str(i+last_code_idx+1)+'\n')
            tbar.set_description(f"Curr succ rate {succ_times/(i+last_code_idx+1):3f}")  
        self.logger.info(f"[Task Done] Time Cost: {time.time()-st_time:1f} sec Succ Rate: {succ_times/data_len:.3f}\n")
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
    parser.add_argument('--attack_model',type=str,default='graphcodebert',help="support lstm,gru")
    parser.add_argument('--attack_way',type=str,default='pretrain',help="support attack ways [grad_rename,random_rename] ")
    parser.add_argument('--attack_data',type=str,default='test',help="Attack dataset")
    parser.add_argument("--pretrain_model", default="microsoft/codebert-base-mlm", type=str,
                        help="Base Model microsoft/codebert-base-mlm or microsoft/graphcodebert-base") 
    parser.add_argument("--block_size", default=512, type=int,
                        help="input len")
    
    parser.add_argument('--log_path',type=str,default=None,help='Specially specified debug log path')
    parser.add_argument('--error_log_path',type=str,default='../save/attack/record_problem.log')
    parser.add_argument('--sample_nums',type=int,default=-1,help="Only the sampled data is attacked. If it is less than 0, it means that it is not sampled")
    
    args=parser.parse_args()
    attack_model_name=args.attack_model
    if attack_model_name == 'codebert':
        args.pretrain_model="microsoft/codebert-base-mlm"
    elif attack_model_name == 'graphcodebert':
        args.pretrain_model="microsoft/graphcodebert-base"
    attack_way=args.attack_way
    attacl_data=args.attack_data
    if attack_model_name in support_model.keys() and attack_way in attack_map.keys() :
        
        set_seed()
        mycfg=myConfig()
        config=mycfg.config
        logger=logging.getLogger(__name__)
        model_config=config[attack_model_name]
        attack_config=config[attack_map[attack_way]]
        if attacl_data=='train':
            attack_config['is_enhance_data']=True
        
        if args.log_path is None:
            import datetime
            day=datetime.date.today().day
            log_path = os.path.join(attack_config['log_path'], args.attack_model, args.attack_data+"_"+args.attack_way+"_day"+str(day)+".log")
            args.error_log_path=os.path.join(attack_config['log_path'],args.attack_model+"_"+args.attack_data+"_"+args.attack_way+"_day"+str(day)+".log")
        else:
            log_path=args.log_path
        attack_config['log_path']=log_path
        log_write_mode=attack_config['log_write_mode']
        set_attack_logger(logger,log_path=attack_config['log_path'],write_mode=log_write_mode)
        
        n_gpu,device=set_device(config)
        model_config['device']=device
        vocab = vocab_map[attack_model_name](config=config)
        model_config['vocab']=vocab
        
        model=support_model[attack_model_name](**model_config)
        if n_gpu:
            model.load_state_dict(torch.load(model_config['load_path']))
        else:
            model.load_state_dict(torch.load(model_config['load_path'],map_location='cpu'))
        model.eval() 
       
        target_dataset=support_dataset[attack_model_name](vocab,attacl_data,attack_model_name,config=config)
       
        pretrainA=pretrainAttack(args=args,dataset=target_dataset,model=model,vocab=vocab,device=device,logger=logger,**attack_config)
        file = args.error_log_path
        with open(file, 'a', encoding='utf-8') as fileobj:
            fileobj.write(f"{attack_model_name} problem code record\n")
        pretrainA.attack_dataset()
