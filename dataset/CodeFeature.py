'''
Author: Zhang haha
Date: 2022-07-31 06:03:16
LastEditTime: 2023-01-20
Description: Feature class of code
'''

from turtle import clear
import pycparser
import numpy as np
import sys

sys.path.append("..")
from utils.tools import *
from utils.basic_setting import myConfig
from utils.pattern import *
from vocabClass import VocabModel, VocabModel_ggnn_without_seq,VocabModel_tbcnn,VocabModel_astnn,VocabModel_codebert,VocabModel_graphcodebert 
class CodeFeatures(object):
    """For LSTM and RNN"""
    def __init__(self,code_id,code,label,vocab:VocabModel,max_len=300,rename=False,struct_attack=False):
        self.id=code_id
        self.code=code#
        self.label=label
        self.vocab=vocab
        self.max_len=max_len
        self.tokens=vocab.getCodeTokens(code)
        self.tokens_idx=vocab.tokenize(self.tokens)
        self.lens=len(self.tokens) 
        self.input_idx=self._trim(self.tokens_idx,max_len,vocab['<pad>'])
        self.mask=self.get_mask()
        
        if rename:
        # prepare for node attack相
            self.prepare_for_rename()
        if struct_attack:
        # prepare for struckt attack
            self.prepare_for_struct_attack()
    def prepare_for_rename(self,vocab):
        self.tokens_with_pos,self.tokens_with_pos_all=self.get_target_tokens(vocab)
    def prepare_for_struct_attack(self):
        self.stmt_attack_poses,self.stmt_attack_level,self.tokens=StmtInsPos(self.tokens)
        self.code=tokens2code(self.tokens)
        
    def get_mask(self):
        mask=np.zeros(self.max_len)
        mask[:self.lens]=True
        return mask
    def get_target_tokens(self,vocab):
        target_tokens=self.vocab.get_code_candidate_tokens(self.code,vocab.valid_tokens)
        tokens_with_pos={}
        tokens_with_pos_all={}
        for i in range(len(self.tokens)):
            if self.tokens[i] in target_tokens:
                if self.tokens[i] in tokens_with_pos.keys() and i <self.max_len:
                    tokens_with_pos[self.tokens[i]].append(i)
                if self.tokens[i] in tokens_with_pos_all.keys():
                    tokens_with_pos_all[self.tokens[i]].append(i)
                else:
                    if i < self.max_len:
                        tokens_with_pos[self.tokens[i]]=[i]
                    tokens_with_pos_all[self.tokens[i]]=[i]
        return tokens_with_pos,tokens_with_pos_all
    def _trim(self,tokens_idx,max_len,pad):
        
        if len(tokens_idx)>max_len:
            return tokens_idx[:max_len]
        else:
            return tokens_idx+[pad]*(max_len-len(tokens_idx)) 
    
class CodeFeatures_tbcnn(object):
    """for tbcnn"""
    def __init__(self,code_id,code,label,vocab:VocabModel_tbcnn,rename=False,struct_attack=False):
        self.id=code_id
        self.code=code
        self.label=label
        self.vocab=vocab
        self.tokens=vocab.getCodeTokens(code)
        self.lens=len(self.tokens) 
        self.nodes,self.begins, self.ends=vocab.get_tbcnn_graph(self.code)
        self.graph=self.vocab.get_pyg_graph( self.nodes,self.begins, self.ends,self.label)
        self.nodes_len=len(self.nodes)    
        
        if rename:
            self.prepare_for_rename()
        if struct_attack:
            self.prepare_for_struct_attack()
    def prepare_for_rename(self,valid_tokens=None):
        self.tokens_with_pos_all=self.get_target_tokens(valid_tokens)
    def prepare_for_struct_attack(self):
        self.stmt_attack_poses,self.stmt_attack_level,self.tokens=StmtInsPos(self.tokens)
        self.code=" ".join(self.tokens)
    def get_target_tokens(self,valid_tokens=None):
        if valid_tokens is None:
            valid_tokens=self.vocab.all_candidate_tokens
        target_tokens=self.vocab.get_code_candidate_tokens(self.code,valid_tokens)
        tokens_with_pos_all={}
        for i in range(len(self.tokens)): 
            if self.tokens[i] in target_tokens:
                if self.tokens[i] in tokens_with_pos_all.keys():
                    tokens_with_pos_all[self.tokens[i]].append(i)
                else:
                    tokens_with_pos_all[self.tokens[i]]=[i]
        return tokens_with_pos_all

class CodeFeatures_astnn(object):
    """for astnn"""
    def __init__(self,code_id,code,label,vocab:VocabModel_astnn,rename=False,struct_attack=False):
        self.id=code_id
        self.code=code
        self.label=label
        self.tokens=vocab.getCodeTokens(code)
        self.lens=len(self.tokens) 
        self.tree_in=vocab.trans2input(code)   
        if rename:
            self.prepare_for_rename(vocab)
        if struct_attack:
            self.prepare_for_struct_attack()
    def prepare_for_rename(self,vocab):
        self.tokens_with_pos_all=self.get_target_tokens(vocab)
    def prepare_for_struct_attack(self):
        self.stmt_attack_poses,self.stmt_attack_level,self.tokens=StmtInsPos(self.tokens)
        self.code=tokens2code(self.tokens)
    def get_target_tokens(self,vocab):
        target_tokens=set(vocab.get_code_candidate_tokens(self.code,vocab.valid_tokens))
        target_tokens=list(target_tokens-set(vocab.forbidden_tokens))
        
        tokens_with_pos_all={}
        for i in range(len(self.tokens)):
            if self.tokens[i] in target_tokens:
                if self.tokens[i] in tokens_with_pos_all.keys():
                    tokens_with_pos_all[self.tokens[i]].append(i)
                else:
                    tokens_with_pos_all[self.tokens[i]]=[i]
        return tokens_with_pos_all

class CodeFeatures_codebert(object):
    """for codebert"""
    def __init__(self,code_id,code,label,vocab:VocabModel_codebert,rename=False,struct_attack=False):
        self.id=code_id
        self.label=label
        self.tokens=vocab.getCodeTokens(code) 
        self.code=" ".join(self.tokens)
        self.input_ids=vocab.tokenize(self.code,cut_and_pad=True,ret_id=True)[0]

        
        if rename:
            self.prepare_for_rename(vocab)
        if struct_attack:
            self.prepare_for_struct_attack(vocab)
    def prepare_for_rename(self,vocab):
        self.target_subtokens=self.get_target_tokens(vocab)
    def prepare_for_struct_attack(self):
        self.stmt_attack_poses,self.stmt_attack_level,self.tokens=StmtInsPos(self.tokens)
        self.code=tokens2code(self.tokens)
    def get_target_tokens(self,vocab:VocabModel_codebert):
        self.target_tokens=VocabModel.get_code_candidate_tokens(self.code)
        if len(self.target_tokens)==0:
            return []
        target_subtokens=vocab.tokenizer.tokenize(" ".join(self.target_tokens))
        claer_target_subtokens=['Ġ'+target_subtokens[0]]
        claer_target_subtokens=claer_target_subtokens+[subtoken for subtoken in target_subtokens[1:] if subtoken.startswith('Ġ')]
        claer_target_subtokens=set(claer_target_subtokens)&set(vocab.all_candidate_tokens)
        return list(claer_target_subtokens)
        
        
class CodeFeatures_graphcodebert(object):
    """for graphcodebert"""
    def __init__(self,code_id,code,label,vocab:VocabModel_graphcodebert,rename=False,struct_attack=False,config=myConfig().config):
        self.id=code_id
        self.tokens=vocab.getCodeTokens(code) 
        self.code=" ".join(self.tokens)
        self.label=label
        self.input_tokens,self.input_ids,self.position_idx,self.dfg_to_code,self.dfg_to_dfg=vocab.convert_examples_to_features(code,vocab.parser,vocab.tokenizer,"c",config['graphcodebert'])
        

        
        if rename:
            self.prepare_for_rename(vocab)
        if struct_attack:
            self.prepare_for_struct_attack(vocab)
    def prepare_for_rename(self,vocab):
        self.target_subtokens=self.get_target_tokens(vocab)
    def prepare_for_struct_attack(self):
        try:
            self.stmt_attack_poses,self.stmt_attack_level,self.tokens=StmtInsPos(self.tokens)
            self.code=tokens2code(self.tokens)
        except:
            self.tokens=VocabModel.getCodeTokens(self.code)
            self.stmt_attack_poses,self.stmt_attack_level,self.tokens=StmtInsPos(self.tokens)
            self.code=tokens2code(self.tokens)

    def get_target_tokens(self,vocab:VocabModel_codebert):
        self.target_tokens=VocabModel.get_code_candidate_tokens(self.code)
        if len(self.target_tokens)==0:
            return []
        target_subtokens=vocab.tokenizer.tokenize(" ".join(self.target_tokens))
        claer_target_subtokens=['Ġ'+target_subtokens[0]]
        claer_target_subtokens=claer_target_subtokens+[subtoken for subtoken in target_subtokens[1:] if subtoken.startswith('Ġ')]
        claer_target_subtokens=set(claer_target_subtokens)&set(vocab.all_candidate_tokens)
        return list(claer_target_subtokens)
        
class CodeFeatures_ggnn(object):
    """for ggnn"""
    def __init__(self,code_id,code,label,vocab,rename=False,struct_attack=False):
        self.id=code_id
        self.tokens=vocab.getCodeTokens(code) 
        self.code=" ".join(self.tokens)
        self.label=label
        self.graph=vocab.code_to_graph(code)        

        if rename:
        
            self.prepare_for_rename(vocab)
        if struct_attack:
            self.prepare_for_struct_attack()
    def prepare_for_rename(self,vocab:VocabModel_ggnn_without_seq):
        self.target_subtokens=self.get_target_tokens(vocab)
    def prepare_for_struct_attack(self):
        self.stmt_attack_poses,self.stmt_attack_level,self.tokens=StmtInsPos(self.tokens)
        self.code=tokens2code(self.tokens)
    def get_target_tokens(self,vocab:VocabModel_ggnn_without_seq):
        self.target_tokens=VocabModel.get_code_candidate_tokens(self.code)
        target_subtokens=vocab.tokenize_idxs2tokens(vocab.tokenize_tokens2idxs(self.target_tokens))
        target_subtokens=set(target_subtokens)&set(vocab.all_candidate_tokens)
        
        return list(target_subtokens)
    def get_node_nums(self):
        return len(self.graph['nodes'])
    def get_edge_nums(self):
        return len(self.graph['edges'])
    def get_token_length(self):
        return len(self.graph['backbone_sequence'])    


if __name__=='__main__':
    code="""
    void main ( ) { int j , i , max = 0 ; char str [ 11 ] = { 0 } , substr [ 4 ] = { 0 } , s [ 20 ] = { 0 } , * p ; while ( scanf ( "%s %s" , str , substr ) != EOF ) { max = str [ 0 ] ; j = 0 ; p = & str [ 0 ] ; for ( i = 0 ; str [ i ] != 0 ; i ++ ) if ( str [ i ] > max ) { max = str [ i ] ; p = & str [ i ] ; j = i ; } strncat ( s , str , j + 1 ) ; strcat ( s , substr ) ; strcat ( s , p + 1 ) ; printf ( "%s\\n" , s ) ; strcpy ( s , "" ) ; strcpy ( str , "" ) ; strcpy ( substr , "" ) ; } }
    """
    config=myConfig().config
    vocab=VocabModel_ggnn_without_seq(config=config)
    codefeatures=CodeFeatures_ggnn(-1,code,1,vocab,struct_attack=True)
    print(codefeatures.stmt_attack_poses)
    from utils.pattern import *
    InsVis(codefeatures.tokens,codefeatures.stmt_attack_poses)