'''
Author: Zhang haha
Date: 2022-07-31 08:38:56
LastEditTime: 2023-01-22
Description: Dictionary class token=>id

'''





from logging import exception
import sys
from typing import Counter

import numpy
from tree_sitter import Language, Parser
sys.path.append("..")
from utils.parser.DFG_c import DFG_c
from utils.parser.utils import *
from utils.tools import read_pkl,save_pkl,const_norm
from utils.basic_setting import *
from utils.generate_graph.ast_graph_generator import CAstGraphGenerator
from data.vocab.forbidden import forbidden_tokens,C_ast_key_words
from data.vocab.random_insert import inserts
import pycparser
import  pycparser.c_ast as C_ast
import torch
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from pycparser.c_parser import CParser
class VocabModel(object):
    
    def __init__(self,config=myConfig.get_config(),vocab_restore=False,canditate_restore=False,logger=None) -> None:        
        '''
        description: Build vocab from the input dataset
        
        '''
        self.vocab_size=config['vocab_set']['vocab_size']
        self.reserved_tokens=["<pad>","<unk>"]
        self.config=config
        self.vocab_restore=vocab_restore
        self.canditate_restore=canditate_restore
        self.logger=logger
        
        self.build_vocab()
       
        self.valid_tokens=set(self.idx2token)
        self.build_all_candidate_tokens()
        self.forbidden_tokens=forbidden_tokens
        self.all_candidate_idxs=self.tokenize(self.all_candidate_tokens)
        self.reserved_tokens_indxs=self.tokenize(self.reserved_tokens)
        
        self.random_attack=[insert.split(" ") for insert in inserts]
    def build_vocab(self):
        
        
        if os.path.isfile(self.config['vocab_set'].get('load_path')) and not self.vocab_restore:
            vocab=read_pkl(self.config['vocab_set']['load_path'])
            if vocab.get('vocab_cnt') is not None:
                self.vocab_cnt=vocab['vocab_cnt']
            self.token2idx=vocab['token2idx']
            self.idx2token=vocab['idx2token']
        else:
            if self.logger is not None:
                self.logger.info(f"building vocab from {self.config['raw_data_path']['train']}")
            self.idx2token=self.reserved_tokens[:]
            self.token2idx=dict(zip(self.reserved_tokens, range(len(self.reserved_tokens))))
            self.vocab_cnt = Counter()
            train_raw=read_pkl(self.config['raw_data_path']['train'])
            self.build_vocab_from_codes(train_raw['code'],self.vocab_size)
            vocab={'vocab_cnt':self.vocab_cnt,'token2idx':self.token2idx,'idx2token':self.idx2token}
            save_pkl(vocab,self.config['vocab_set']['load_path'])
            if self.logger is not None:
                self.logger.info(f"saving vocab in {self.config['vocab_set']['load_path']}")
    def build_vocab_from_codes(self,codes,vocab_size=5000):
        '''
        description: 创建vocab
        param codes [*]原始的以字符串形式存储的code
        param vocab_size [*]词典的大小
        '''
        
        for code in tqdm(codes):
            tokens=self.getCodeTokens(code)
            for token in tokens:
                self.vocab_cnt.update([self.getToken(token)])
        self._add_words(self.vocab_cnt.keys())
        self._trim(vocab_size)
    def _add_words(self,words):
        
        for word in words:
            token=self.getToken(word)
            if token not in self.token2idx:
                self.token2idx[token]=len(self.idx2token)
                self.idx2token.append(token)
    def _trim(self,vocab_size=5000,min_freq=1):
        
        if min_freq <= 1 and (vocab_size is None or vocab_size >= len(self.token2idx)):
            return
        ordered_words = sorted(((c, w) for (w, c) in self.vocab_cnt.items()), reverse=True)
        if vocab_size:
            ordered_words = ordered_words[:vocab_size-len(self.reserved_tokens)]
        self.idx2token = self.reserved_tokens[:]
        self.token2idx = dict(zip(self.reserved_tokens, range(len(self.reserved_tokens))))
        self.vocab_cnt = Counter()
        for count, word in ordered_words:
            if count < min_freq: break
            token=self.getToken(word)
            if token not in self.token2idx:
                self.token2idx[token] = len(self.idx2token)
                self.vocab_cnt[token] = count
                self.idx2token.append(token)
        assert len(self.idx2token) == len(self.token2idx)    
    def __getitem__(self, item):
        
        if type(item) in [int, numpy.int64]:
            if item<self.vocab_size:
                return self.idx2token[item]
            else:
                return "<unk>"
        else:
            token=self.getToken(item)
            if token in self.token2idx.keys():
                return self.token2idx[token]
            else:
                return self.token2idx["<unk>"]
    def tokenize(self,tokens):
        
        if isinstance(tokens,str):
            tokens=self.getCodeTokens(tokens)
        idxs=[]
        for token in tokens:
            idxs.append(self.__getitem__(token))   
        return idxs     
    def __len__(self):
        
        return len(self.idx2token)
    def build_all_candidate_tokens(self):
        
        
        if os.path.isfile(self.config['node'].get('all_candidate_rnn_tokens')) and not self.canditate_restore:
            self.all_candidate_tokens=read_pkl(self.config['node']['all_candidate_rnn_tokens'])
        else:
            if self.logger is not None:
                self.logger.info(f"building all candidate tokens from {self.config['raw_data_path']['train']}")
            old_candidate_tokens=self._build_all_candidate_tokens()
            
            candidate_tokens=set(old_candidate_tokens)&set(self.idx2token)
            candidate_tokens=candidate_tokens-set(self.reserved_tokens)
            
            self.all_candidate_tokens=list(candidate_tokens)
            save_pkl(self.all_candidate_tokens,self.config['node']['all_candidate_rnn_tokens'])
    def _build_all_candidate_tokens(self):
        if os.path.isfile(self.config['node'].get('all_candidate_tokens')) :
            tokens=read_pkl(self.config['node']['all_candidate_tokens'])
            return tokens
        train_raw=read_pkl(self.config['raw_data_path']['train'])
        codes=train_raw['code']
        tokens=set()
        for code in tqdm(codes):
            tokens.update(VocabModel.get_code_candidate_tokens(code,None))
        
        tokens=tokens-set(forbidden_tokens)
        tokens=list(tokens)
        save_pkl(tokens,self.config['node']['all_candidate_tokens'])
        return list(tokens)
    def getToken2idx(self):
        return self.token2idx
    def getIdx2Token(self):
        return self.idx2token
    def get_valid_tokens(self):
        return self.valid_tokens    
    def getToken(self,token):
        
        if token[0] == '"' and token[-1] == '"':
                return "<str>"
        elif token[0] == "'" and token[-1] == "'":
            return "<char>"
        elif token[0] in "0123456789.":
            if 'e' in token.lower():
                return "<fp>" 
            elif '.' in token:
                if token == '.':
                    return token
                else:
                    return "<fp>"
            else:
                return "<int>"
        else:
            return token
    @staticmethod
    def getCodeTokens(code):        
        tokens=[]
        parser = pycparser.CParser()
        parser.clex.input(code)
        t = parser.clex.token()
        while t is not None:
            tokens.append(t.value)
            t = parser.clex.token()
        return tokens
    @classmethod
    def get_code_candidate_tokens(cls,code,valid_tokens=None):

        
        ast=pycparser.CParser().parse(code)
        candidate_tokens=cls.get_ast_candidate_tokens(ast)
        
        if valid_tokens is not None:
            filter_candidate_tokens=set(valid_tokens)&set(candidate_tokens)-set(forbidden_tokens)
            return list(filter_candidate_tokens)
        else: 
            candidate_tokens=candidate_tokens-set(forbidden_tokens)
            return list(candidate_tokens)   
    @classmethod
    def get_ast_candidate_tokens(cls,ast):
        
        token = []
        if isinstance(ast, pycparser.c_ast.Decl) and ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast,pycparser.c_ast.Typedef) and  ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast, pycparser.c_ast.Struct) and ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast, pycparser.c_ast.Enum) and ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast, pycparser.c_ast.Union) and ast.name is not None:
            token.append(ast.name)
        token = set(token) 
        for c in ast.children():
            token.update(cls.get_ast_candidate_tokens(c[1]))
        return token
    @staticmethod
    def get_target_tokens_nums(code_tokens,target_tokens):
        tokens_nums={}
        for token in code_tokens:
            if token in target_tokens:
                try:
                    tokens_nums[token]+=1
                except:
                    tokens_nums[token]=1
        return tokens_nums
class VocabModel_tbcnn(VocabModel):
    
    def __init__(self,config=myConfig.get_config(),vocab_restore=False,canditate_restore=False,logger=None) -> None:        
    
        self.vocab_size=config['vocab_set']['tbcnn_vocab_size']
        self.reserved_tokens=["<unk>"]+C_ast_key_words# 保留token
        
        self.config=config
        self.vocab_restore=vocab_restore
        self.canditate_restore=canditate_restore
        self.logger=logger
        self.parser=pycparser.c_parser.CParser()
        
        self.build_vocab()
        
        self.valid_tokens=set(self.idx2token)
        self.build_all_candidate_tokens()
        
        self.all_candidate_idxs=self.tokenize(self.all_candidate_tokens)
        
        self.random_attack=[insert.split(" ") for insert in inserts]
        
    def build_vocab(self):
        
        if os.path.isfile(self.config['vocab_set'].get('tbcnn_load_path')) and not self.vocab_restore:
            vocab=read_pkl(self.config['vocab_set']['tbcnn_load_path'])
            if vocab.get('vocab_cnt') is not None:
                self.vocab_cnt=vocab['vocab_cnt']
            self.token2idx=vocab['token2idx']
            self.idx2token=vocab['idx2token']
        else:
            if self.logger is not None:
                self.logger.info(f"building vocab from {self.config['raw_data_path']['train']}")
            self.idx2token=self.reserved_tokens[:]
            self.token2idx=dict(zip(self.reserved_tokens, range(len(self.reserved_tokens))))
            self.vocab_cnt = Counter()
            train_raw=read_pkl(self.config['raw_data_path']['train'])
            self.build_vocab_from_codes(train_raw['code'],self.vocab_size)
            vocab={'vocab_cnt':self.vocab_cnt,'token2idx':self.token2idx,'idx2token':self.idx2token}
            save_pkl(vocab,self.config['vocab_set']['tbcnn_load_path'])
            if self.logger is not None:
                self.logger.info(f"saving vocab in {self.config['vocab_set']['tbcnn_load_path']}")
    def build_vocab_from_codes(self,codes,vocab_size=5000):
       
        for code in tqdm(codes):
            nodes,begin_ids,end_ids=self.get_tbcnn_graph(code)
            for node in nodes:
                self.vocab_cnt.update([self.getToken(node)])
        self._add_words(self.vocab_cnt.keys())
        self._trim(vocab_size)
    def _add_words(self,words):
        
        for word in words:
            token=self.getToken(word)
            if token not in self.token2idx:
                self.token2idx[token]=len(self.idx2token)
                self.idx2token.append(token)
    def _trim(self,vocab_size=5000,min_freq=1):
        
        if min_freq <= 1 and (vocab_size is None or vocab_size >= len(self.token2idx)):
            return
        ordered_words = sorted(((c, w) for (w, c) in self.vocab_cnt.items()), reverse=True)
        if vocab_size:
            ordered_words = ordered_words[:vocab_size-len(self.reserved_tokens)]
        self.idx2token = self.reserved_tokens[:]
        self.token2idx = dict(zip(self.reserved_tokens, range(len(self.reserved_tokens))))
        self.vocab_cnt = Counter()
        for count, word in ordered_words:
            if count < min_freq: break
            token=self.getToken(word)
            if token not in self.token2idx:
                self.token2idx[token] = len(self.idx2token)
                self.vocab_cnt[token] = count
                self.idx2token.append(token)
        assert len(self.idx2token) == len(self.token2idx)    
    def __getitem__(self, item):
        
        if type(item) in [int, numpy.int64]:
            if item<self.vocab_size:
                return self.idx2token[item]
            else:
                return "<unk>"
        else:
            token=self.getToken(item)
            if token in self.token2idx.keys():
                return self.token2idx[token]
            else:
                return self.token2idx["<unk>"]
    def tokenize(self,tokens):
        
        if isinstance(tokens,str):
            tokens=self.getCodeTokens(tokens)
        idxs=[]
        for token in tokens:
            idxs.append(self.__getitem__(token))   
        return idxs 
    def __len__(self):
        
        return len(self.idx2token)
    def build_all_candidate_tokens(self):
        
       
        if os.path.isfile(self.config['node'].get('all_candidate_tokens')) and not self.canditate_restore:
            self.all_candidate_tokens=read_pkl(self.config['node']['all_candidate_tbcnn_tokens'])
        else:
            if self.logger is not None:
                self.logger.info(f"building all candidate tokens from {self.config['raw_data_path']['train']}")
            train_raw=read_pkl(self.config['raw_data_path']['train'])
            codes=train_raw['code']
            tokens=set()
            for code in tqdm(codes):
                tokens.update(self.get_code_candidate_tokens(code,self.valid_tokens))
            tokens=tokens&set(self.idx2token)
            tokens=tokens-set(forbidden_tokens)-set(self.reserved_tokens)
            self.all_candidate_tokens=list(tokens)
            save_pkl(self.all_candidate_tokens,self.config['node']['all_candidate_tbcnn_tokens'])
      
    def getToken(self,token):
        
        if token[0] == '"' and token[-1] == '"':
                return "<str>"
        elif token[0] == "'" and token[-1] == "'":
            return "<char>"
        elif token[0] in "0123456789.":
            if 'e' in token.lower():
                return "<fp>" 
            elif '.' in token:
                if token == '.':
                    return token
                else:
                    return "<fp>"
            else:
                return "<int>"
        else:
            return token
    @staticmethod
    def getCodeTokens(code):        
        
        tokens=[]
        parser = pycparser.CParser()
        parser.clex.input(code)
        t = parser.clex.token()
        while t is not None:
            tokens.append(t.value)
            t = parser.clex.token()
        return tokens
    @classmethod
    def get_code_candidate_tokens(cls,code,valid_tokens=None):
        
        ast=pycparser.CParser().parse(code)
        candidate_tokens=cls.get_ast_candidate_tokens(ast)
       
        if not isinstance(valid_tokens,set):
            valid_tokens=set(valid_tokens)
        filter_candidate_tokens=valid_tokens&set(candidate_tokens)
        return list(filter_candidate_tokens)   
    @classmethod
    def get_ast_candidate_tokens(cls,ast):
       
        token = []
        if isinstance(ast, pycparser.c_ast.Decl) and ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast, pycparser.c_ast.Struct) and ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast,pycparser.c_ast.Typedef) and  ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast, pycparser.c_ast.Enum) and ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast, pycparser.c_ast.Union) and ast.name is not None:
            token.append(ast.name)
        token = set(token) 
        for c in ast.children():
            token.update(cls.get_ast_candidate_tokens(c[1]))
        return token    
    @classmethod
    def get_tbcnn_graph(cls,code,parser=pycparser.c_parser.CParser()):
        
        ast=parser.parse(code)
        nodes,begin_ids,end_ids=cls.extract_ast(ast, 0, [SimpleNode(ast).get_token()], [], [])
        return nodes,begin_ids,end_ids
    @classmethod
    def extract_ast(cls,root, idx, nodes, begins, ends):
       
        for s, c in root.children():
            nodes.append(SimpleNode(c).get_token())
            begins.append(len(nodes)-1)
            ends.append(idx)
            nodes, begins, ends = cls.extract_ast(c, begins[-1], nodes, begins, ends)
        return nodes, begins, ends
    
    def get_pyg_graph(self,nodes,begins_ids,end_ids,label):
        
        nodes=self.tokenize(tokens=nodes)# tokens=>ids
        nodes,begins_ids,end_ids=torch.tensor(nodes),torch.tensor(begins_ids),torch.tensor(end_ids)
        label=torch.tensor(label)
        return Data(x=nodes,y=label,edge_index=torch.stack((begins_ids,end_ids),dim=0))
        
class SimpleNode(object):
    
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()

    def is_leaf(self):
        
        if self.is_str:#
            return True
        return len(self.node.children()) == 0

    def get_token(self, lower=True):
        
        if self.is_str:
            return self.node
        name = self.node.__class__.__name__
        token = name
        is_name = False
        if self.is_leaf():
            attr_names = self.node.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = self.node.names[0]
                elif 'name' in attr_names:
                    token = self.node.name
                    is_name = True
                else:
                    token = self.node.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = self.node.declname
            if self.node.attr_names:
                attr_names = self.node.attr_names
                if 'op' in attr_names:
                    if self.node.op[0] == 'p':
                        token = self.node.op[1:]
                    else:
                        token = self.node.op
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
            
        return const_norm(token)

class ASTNode(SimpleNode):
   
    def __init__(self, node):
        super().__init__(node)
        self.children = self.add_children()
    def add_children(self):
        
        if self.is_str:
            return []
        children = self.node.children()
        if self.token in ['FuncDef', 'If', 'While', 'DoWhile','Switch']:
            return [ASTNode(children[0][1])]
        elif self.token == 'For':
            return [ASTNode(children[c][1]) for c in range(0, len(children)-1)]
        else:
            return [ASTNode(child) for _, child in children]

    def children(self):
        return self.children
class VocabModel_astnn(VocabModel):
    
    def __init__(self,config=myConfig.get_config(),vocab_restore=False,canditate_restore=False,logger=None) -> None:        
        
        self.config=config
        self.vocab_restore=vocab_restore
        self.canditate_restore=canditate_restore
        self.logger=logger
        self.parser=pycparser.c_parser.CParser()
        
        
        self.build_vocab()
        
        
        self.valid_tokens=set(self.idx2token)
        self.build_all_candidate_tokens()
        self.forbidden_tokens=forbidden_tokens
        self.all_candidate_idxs=self.tokenize(self.all_candidate_tokens)
       
        self.random_attack=[insert.split(" ") for insert in inserts]# 可以随机插入的dead code
        
    def build_vocab(self):
        
        if os.path.isfile(self.config['vocab_set'].get('astnn_load_path')) and not self.vocab_restore:
            self.word2vec=Word2Vec.load(self.config['vocab_set']['astnn_load_path']).wv 
        else:
            if self.logger is not None:
                self.logger.info(f"building astnn vocab from {self.config['raw_data_path']['train']}")
            train_raw=read_pkl(self.config['raw_data_path']['train'])
            word2vec=self.build_vocab_from_codes(train_raw['code'])
            word2vec.save(self.config['vocab_set']['astnn_load_path'])
            self.word2vec=word2vec.wv
            if self.logger is not None:
                self.logger.info(f"saving vocab in {self.config['vocab_set']['astnn_load_path']}_w2v_{str(self.config['astnn']['word2vec']['vector_size'])}")
        self.token2idx=self.word2vec.key_to_index
        self.idx2token=self.word2vec.index_to_key
        self.vocab_size=self.word2vec.vectors.shape[0]+1
        
        self.embeddings = np.zeros((self.vocab_size , self.word2vec.vectors.shape[1]), dtype="float32")     
        self.embeddings[:self.vocab_size-1] = self.word2vec.vectors
        
    def build_vocab_from_codes(self,codes):
        
        corpus=[]
        for code in tqdm(codes):
            
            ast=self.parser.parse(code)
            corpus.append(self.trans2sequences(ast))
        
            
        word2vec = Word2Vec(corpus, **self.config['astnn']['word2vec'])
        return word2vec
    def __getitem__(self, item):
        
        if type(item) in [int, numpy.int64]:
            if item<self.vocab_size-1:
                return self.idx2token[item]
            else:
                return "<unk>"
        else:
            if item in self.token2idx.keys():
                return self.token2idx[item]
            else:
                return self.vocab_size-1
    def tokenize_tree(self, node):
        
        token = node.token
        result = [self.__getitem__(token)]
        children = node.children
        for child in children:
            result.append(self.tokenize_tree(child))
        return result
    def tokenize(self,tokens):
       
        if isinstance(tokens,str):
            tokens=self.getCodeTokens(tokens)
        idxs=[]
        for token in tokens:
            idxs.append(self.__getitem__(token))   
        return idxs   
    def __len__(self):
        
        return len(self.idx2token)
    def build_all_candidate_tokens(self):
       
       
        if os.path.isfile(self.config['node'].get('all_candidate_astnn_tokens')) and not self.canditate_restore:
            self.all_candidate_tokens=read_pkl(self.config['node']['all_candidate_astnn_tokens'])
        else:
            if self.logger is not None:
                self.logger.info(f"building all candidate tokens from {self.config['raw_data_path']['train']}")
            train_raw=read_pkl(self.config['raw_data_path']['train'])
            codes=train_raw['code']
            tokens=set()
            for code in tqdm(codes):
                tokens.update(self.get_code_candidate_tokens(code,self.valid_tokens))
            tokens=tokens&set(self.idx2token)
            tokens=tokens-set(forbidden_tokens)
            self.all_candidate_tokens=list(tokens)
            save_pkl(self.all_candidate_tokens,self.config['node']['all_candidate_astnn_tokens'])
      
    @staticmethod
    def getCodeTokens(code):        
        
        tokens=[]
        parser = pycparser.CParser()
        parser.clex.input(code)
        t = parser.clex.token()
        while t is not None:
            tokens.append(t.value)
            t = parser.clex.token()
        return tokens
    @classmethod
    def get_code_candidate_tokens(cls,code,valid_tokens=None):
       
       
        ast=pycparser.CParser().parse(code)
        candidate_tokens=cls.get_ast_candidate_tokens(ast)
       
        if valid_tokens is not None:
            filter_candidate_tokens=valid_tokens&set(candidate_tokens)
            return list(filter_candidate_tokens)
        else: return candidate_tokens   
    @classmethod
    def get_ast_candidate_tokens(cls,ast):
       
        token = []
        if isinstance(ast, pycparser.c_ast.Decl) and ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast,pycparser.c_ast.Typedef) and  ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast, pycparser.c_ast.Struct) and ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast, pycparser.c_ast.Enum) and ast.name is not None:
            token.append(ast.name)
        elif isinstance(ast, pycparser.c_ast.Union) and ast.name is not None:
            token.append(ast.name)
        token = set(token) 
        for c in ast.children():
            token.update(cls.get_ast_candidate_tokens(c[1]))
        return token    
    @classmethod
    def get_sequences(cls,node,sequence):
       
        current = ASTNode(node) 
        sequence.append(current.get_token())
        for _, child in node.children():
            cls.get_sequences(child, sequence) 
        if current.get_token().lower() == 'compound':
            sequence.append('End')
            
    @classmethod
    def trans2sequences(cls,ast):
        sequence = []
        cls.get_sequences(ast, sequence)
        return sequence
    @classmethod
    def get_blocks(cls,node, block_seq):
        
        children = node.children()
        name = node.__class__.__name__
        if name in ['FuncDef', 'If', 'For', 'While', 'DoWhile']:
            block_seq.append(ASTNode(node))
            if name != 'For':
                skip = 1
            else:
                skip = len(children) - 1

            for i in range(skip, len(children)):
                child = children[i][1]
                if child.__class__.__name__ not in ['FuncDef', 'If', 'For', 'While', 'DoWhile', 'Compound']:
                    block_seq.append(ASTNode(child))
                cls.get_blocks(child, block_seq)
        elif name == 'Compound':
            block_seq.append(ASTNode(name))
            for _, child in node.children():
                if child.__class__.__name__ not in ['If', 'For', 'While', 'DoWhile']:
                    block_seq.append(ASTNode(child)) 
                cls.get_blocks(child, block_seq)
            block_seq.append(ASTNode('End'))
        else:
            for _, child in node.children():
                cls.get_blocks(child, block_seq)
   
    def trans2input(self,ast):
        
        if isinstance(ast,str):
            ast=self.parser.parse(ast)
        blocks = []
        self.get_blocks(ast, blocks)
        tree = []
        for b in blocks:
            btree = self.tokenize_tree(b)
            tree.append(btree)
        return tree

from transformers import RobertaTokenizer
class VocabModel_codebert(object):
    
    def __init__(self,canditate_restore=False,logger=None,config=myConfig.get_config()) -> None:        
        
        
        self.canditate_restore=canditate_restore
        self.logger=logger
        self.vocab_size=config['codebert']['vocab_size']
        self.input_len=config['codebert']['input_len']
        self.config=config
        self.tokenizer = RobertaTokenizer.from_pretrained(config['codebert']['model_name'])
       
        
       
        self.forbidden_tokens=forbidden_tokens
        self.build_all_candidate_tokens()
        
        self.all_candidate_idxs=self.tokenizer.convert_tokens_to_ids(self.all_candidate_tokens)
        
        self.random_attack=[insert.split(" ") for insert in inserts]
        
    def __getitem__(self, item):
        
        if type(item) in [int, numpy.int64]:
            if item<self.vocab_size:
                return self.tokenizer._convert_id_to_token(item)
            else:
                return "<unk>"
        else:
            return self.tokenizer._convert_token_to_id(item)
    
        
    def __len__(self):
        
        return self.vocab_size
    
    def tokenize(self,inputs,cut_and_pad=True,ret_id=True):
        
        
        rets = []
        if isinstance(inputs, str):
            inputs = [inputs]
        for sent in inputs:
            if cut_and_pad:
                tokens = self.tokenizer.tokenize(sent)[:self.input_len-2]
                
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                
                padding_length = self.input_len - len(tokens)
                tokens += [self.tokenizer.pad_token] * padding_length
            else:
                
                tokens = self.tokenizer.tokenize(sent)
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            if not ret_id:
                rets.append(tokens)
            else:
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                rets.append(ids)
        return rets

    def build_all_candidate_tokens(self):
        
        
        if os.path.isfile(self.config['node'].get('all_candidate_codebert_tokens')) and not self.canditate_restore:
            self.all_candidate_tokens=read_pkl(self.config['node']['all_candidate_codebert_tokens'])
        else:
            if self.logger is not None:
                self.logger.info(f"building all candidate tokens from {self.config['raw_data_path']['train']} and codebert vocab")        
            old_candidate_tokens=self._build_all_candidate_tokens()
            
            candidate_tokens=[]
            
            
            for subtoken_idx in tqdm(range(self.vocab_size)):
                subtoken = self.tokenizer.convert_ids_to_tokens(subtoken_idx)
                assert isinstance(subtoken, str)
                
                if subtoken in [self.tokenizer.bos_token, self.tokenizer.eos_token,
                                    self.tokenizer.sep_token, self.tokenizer.pad_token,
                                    self.tokenizer.unk_token, self.tokenizer.cls_token,
                                    self.tokenizer.mask_token]:
                    continue
                if not subtoken.startswith('Ġ'):
                    continue
                # Ġxxxx subtoken is the start token of the new word, we only take these subtokens as candidates
                clear_subtoken = subtoken[1:]# 去掉 Ġ
                if clear_subtoken=="":
                    continue
                if clear_subtoken[0] in '0987654321':
                    continue
                
                for token in old_candidate_tokens:
                    
                    if clear_subtoken in token and \
                    clear_subtoken not in candidate_tokens and \
                    clear_subtoken not in self.forbidden_tokens:
                        candidate_tokens.append(clear_subtoken)
                        
                        break
                
            self.all_candidate_tokens=candidate_tokens
            save_pkl(self.all_candidate_tokens,self.config['node']['all_candidate_codebert_tokens'])
    def _build_all_candidate_tokens(self):
       
        
        train_raw=read_pkl(self.config['raw_data_path']['train'])
        codes=train_raw['code']
        tokens=set()
        for code in tqdm(codes):
            tokens.update(VocabModel.get_code_candidate_tokens(code,None))
        
        tokens=tokens-set(forbidden_tokens)
        return list(tokens)

    @staticmethod
    def getCodeTokens(code):        
        
        tokens=[]
        parser = pycparser.CParser()
        parser.clex.input(code)
        t = parser.clex.token()
        while t is not None:
            tokens.append(t.value)
            t = parser.clex.token()
        return tokens
    def getToken(self,token):
        
        if token[0] == '"' and token[-1] == '"':
                return "<str>"
        elif token[0] == "'" and token[-1] == "'":
            return "<char>"
        elif token[0] in "0123456789.":
            if 'e' in token.lower(): 
                return "<fp>" 
            elif '.' in token:
                if token == '.':
                    return token
                else:
                    return "<fp>"
            else:
                return "<int>"
        else:
            return token     

    def get_code_candidate_tokens(self,code):
        
        
        code_subtokens=self.tokenize(code,cut_and_pad=False,ret_id=False)[0]
        code_subtokens=[subtoken[1:] for subtoken in code_subtokens if subtoken.startswith('Ġ')]
        
        try:
            code_subtokens=set(self.all_candidate_tokens)&set(code_subtokens)
            return list(code_subtokens)
        except:
            return code_subtokens   
from transformers import (RobertaForMaskedLM, RobertaTokenizer)
class VocabModel_codebert_enhance(object):
    
    def __init__(self,canditate_restore=False,logger=None,config=myConfig.get_config()) -> None:        
       
        
        self.canditate_restore=canditate_restore
        self.logger=logger
        self.vocab_size=config['codebert']['vocab_size']
        self.input_len=config['codebert']['input_len']
        self.config=config
        self.tokenizer = RobertaTokenizer.from_pretrained(config['codebert']['model_name'])# 加载分词器
        
        self.forbidden_tokens=forbidden_tokens
        self.build_all_candidate_tokens()
        
        self.all_candidate_idxs=self.tokenizer.convert_tokens_to_ids(self.all_candidate_tokens)
        
        self.random_attack=[insert.split(" ") for insert in inserts]
        
    def __getitem__(self, item):
        
        if type(item) in [int, numpy.int64]:
            if item<self.vocab_size:
                return self.tokenizer._convert_id_to_token(item)
            else:
                return "<unk>"
        else:
            return self.tokenizer._convert_token_to_id(item)
    
        
    def __len__(self):
        
        return self.vocab_size
    
    def tokenize(self,inputs,cut_and_pad=True,ret_id=True):
        
        
        rets = []
        if isinstance(inputs, str):
            inputs = [inputs]
        for sent in inputs:
            if cut_and_pad:
                tokens = self.tokenizer.tokenize(sent)[:self.input_len-2]
                
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                
                padding_length = self.input_len - len(tokens)
                tokens += [self.tokenizer.pad_token] * padding_length
            else:
                
                tokens = self.tokenizer.tokenize(sent)
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            if not ret_id:
                rets.append(tokens)
            else:
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                rets.append(ids)
        return rets

    def build_all_candidate_tokens(self):
        
        
        if os.path.isfile(self.config['node'].get('all_candidate_codebert_tokens')) and not self.canditate_restore:
            self.all_candidate_tokens=read_pkl(self.config['node']['all_candidate_codebert_tokens'])
        else:
            if self.logger is not None:
                self.logger.info(f"building all candidate tokens from {self.config['raw_data_path']['train']} and codebert vocab")        
            old_candidate_tokens=self._build_all_candidate_tokens()
            
            candidate_tokens=[]
            
            
            for subtoken_idx in tqdm(range(self.vocab_size)):
                subtoken = self.tokenizer.convert_ids_to_tokens(subtoken_idx)
                assert isinstance(subtoken, str)
                
                if subtoken in [self.tokenizer.bos_token, self.tokenizer.eos_token,
                                    self.tokenizer.sep_token, self.tokenizer.pad_token,
                                    self.tokenizer.unk_token, self.tokenizer.cls_token,
                                    self.tokenizer.mask_token]:
                    continue
                if not subtoken.startswith('Ġ'):
                    continue
                
                clear_subtoken = subtoken[1:]
                if clear_subtoken=="":
                    continue
                if clear_subtoken[0] in '0987654321':
                    continue
                
                for token in old_candidate_tokens:
                    
                    if clear_subtoken in token and \
                    clear_subtoken not in candidate_tokens and \
                    clear_subtoken not in self.forbidden_tokens:
                        candidate_tokens.append(clear_subtoken)
                        
                        break
                
            self.all_candidate_tokens=candidate_tokens
            save_pkl(self.all_candidate_tokens,self.config['node']['all_candidate_codebert_tokens'])
    def _build_all_candidate_tokens(self):
        
        
        train_raw=read_pkl(self.config['raw_data_path']['train'])
        codes=train_raw['code']
        tokens=set()
        for code in tqdm(codes):
            tokens.update(VocabModel.get_code_candidate_tokens(code,None))
        
        tokens=tokens-set(forbidden_tokens)
        return list(tokens)

    @staticmethod
    def getCodeTokens(code):        
       
        tokens=[]
        parser = pycparser.CParser()
        parser.clex.input(code)
        t = parser.clex.token()
        while t is not None:
            tokens.append(t.value)
            t = parser.clex.token()
        return tokens
    def getToken(self,token):
       
        if token[0] == '"' and token[-1] == '"':
                return "<str>"
        elif token[0] == "'" and token[-1] == "'":
            return "<char>"
        elif token[0] in "0123456789.":
            if 'e' in token.lower():
                return "<fp>" 
            elif '.' in token:
                if token == '.':
                    return token
                else:
                    return "<fp>"
            else:
                return "<int>"
        else:
            return token     

    def get_code_candidate_tokens(self,code):
        
        code_subtokens=self.tokenize(code,cut_and_pad=False,ret_id=False)[0]
        code_subtokens=[subtoken[1:] for subtoken in code_subtokens if subtoken.startswith('Ġ')]
        
        try:
            code_subtokens=set(self.all_candidate_tokens)&set(code_subtokens)
            return list(code_subtokens)
        except:
            return code_subtokens       
    

class VocabModel_graphcodebert(object):
   
    def __init__(self,canditate_restore=False,logger=None,config=myConfig.get_config()) -> None:        
        
        
        self.canditate_restore=canditate_restore
        self.logger=logger
        self.vocab_size=config['graphcodebert']['vocab_size']
        self.config=config
        self.tokenizer = RobertaTokenizer.from_pretrained(config['graphcodebert']['model_name'])
        self.parser=self.set_parser()
        
       
        self.forbidden_tokens=forbidden_tokens
        self.build_all_candidate_tokens()
        
        self.all_candidate_idxs=self.tokenizer.convert_tokens_to_ids(self.all_candidate_tokens)
        
        self.random_attack=[insert.split(" ") for insert in inserts]
    def set_parser(self,lang='c'):
        LANGUAGE = Language('../utils/parser/my-languages.so', lang)
        parser = Parser()
        parser.set_language(LANGUAGE)
        parser=[parser,DFG_c]
        return parser
    def __getitem__(self, item):
        
        if type(item) in [int, numpy.int64]:
            if item<self.vocab_size:
                return self.tokenizer._convert_id_to_token(item)
            else:
                return "<unk>"
        else:
            return self.tokenizer._convert_token_to_id(item)
    
        
    def __len__(self):
        
        return self.vocab_size
    def tokenize(self,inputs,ret_id=True):
        
        
        rets = []
        if isinstance(inputs, str):
            inputs = [inputs]
        for sent in inputs:

            tokens = self.tokenizer.tokenize(sent)
            if not ret_id:
                rets.append(tokens)
            else:
                ids = self.tokenizer.convert_tokens_to_ids(tokens)
                rets.append(ids)
        return rets

    def build_all_candidate_tokens(self):
        
        
        if os.path.isfile(self.config['node'].get('all_candidate_graphcodebert_tokens')) and not self.canditate_restore:
            self.all_candidate_tokens=read_pkl(self.config['node']['all_candidate_graphcodebert_tokens'])
        else:
            if self.logger is not None:
                self.logger.info(f"building all candidate tokens from {self.config['raw_data_path']['train']} and graphcodebert vocab")        
            old_candidate_tokens=read_pkl(self.config['node']['all_candidate_tokens'])
            
            candidate_tokens=[]
            
            for subtoken_idx in tqdm(range(self.vocab_size)):
                subtoken = self.tokenizer.convert_ids_to_tokens(subtoken_idx)
                assert isinstance(subtoken, str)
                
                if subtoken in [self.tokenizer.bos_token, self.tokenizer.eos_token,
                                    self.tokenizer.sep_token, self.tokenizer.pad_token,
                                    self.tokenizer.unk_token, self.tokenizer.cls_token,
                                    self.tokenizer.mask_token]:
                    continue
                if not subtoken.startswith('Ġ'):
                    continue
                # Ġxxxx subtoken is the start token of the new word, we only take these subtokens as candidates
                clear_subtoken = subtoken[1:]
                if clear_subtoken=="":
                    continue
                if clear_subtoken[0] in '0987654321':
                    continue
                
                for token in old_candidate_tokens:
                    
                    if clear_subtoken in token and \
                    clear_subtoken not in candidate_tokens and \
                    clear_subtoken not in self.forbidden_tokens:
                        candidate_tokens.append(clear_subtoken)
                        break
                
    
            self.all_candidate_tokens=candidate_tokens
            save_pkl(self.all_candidate_tokens,self.config['node']['all_candidate_graphcodebert_tokens'])
    def getToken(self,token):
       
        if token[0] == '"' and token[-1] == '"':
                return "<str>"
        elif token[0] == "'" and token[-1] == "'":
            return "<char>"
        elif token[0] in "0123456789.":
            if 'e' in token.lower():
                return "<fp>" 
            elif '.' in token:
                if token == '.':
                    return token
                else:
                    return "<fp>"
            else:
                return "<int>"
        else:
            return token            
    @staticmethod
    def getCodeTokens(code):        
        
        tokens=[]
        parser = pycparser.CParser()
        parser.clex.input(code)
        t = parser.clex.token()
        while t is not None:
            tokens.append(t.value)
            t = parser.clex.token()
        return tokens

    def get_code_candidate_tokens(self,code):
       
        
        code_subtokens=self.tokenize(code,ret_id=False)[0]
        code_subtokens=[subtoken[1:] for subtoken in code_subtokens if subtoken.startswith('Ġ')]
        
        filter_candidate_tokens=set(self.all_candidate_tokens)&set(code_subtokens)
        return list(filter_candidate_tokens)
    @staticmethod
    def extract_dataflow(code,parser,lang='c'):
        
        
        if isinstance(code,list):
            code_str = ""
            for t in code:
                code_str += t + " "
            code=code_str
           
       
        try:
            tree = parser[0].parse(bytes(code,'utf8'))    
            root_node = tree.root_node  
            tokens_index=tree_to_token_index(root_node)     
            code=code.split('\n')
            code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
            index_to_code={}
            for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
                index_to_code[index]=(idx,code)  
            try:
                DFG,_=parser[1](root_node,index_to_code,{}) 
            except:
                DFG=[]
            DFG=sorted(DFG,key=lambda x:x[1])
            indexs=set()
            for d in DFG:
                if len(d[-1])!=0:
                    indexs.add(d[1])
                for x in d[-1]:
                    indexs.add(x)
            new_DFG=[]
            for d in DFG:
                if d[1] in indexs:
                    new_DFG.append(d)
            dfg=new_DFG
        except:
            code_toekns=[]
            dfg=[]
        return code_tokens,dfg
    @classmethod
    def convert_examples_to_features(cls,raw_code,parser,tokenizer,lang='c',config=None):
       
        #extract data flow
        code_tokens,dfg=cls.extract_dataflow(raw_code,parser,lang)
        
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]
                
        #truncating
       
        code_tokens=code_tokens[:config['code_length']+config['data_flow_length']-2-min(len(dfg),config['data_flow_length'])]
        # source
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg=dfg[:config['code_length']+config['data_flow_length']-len(source_tokens)]
        source_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        source_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=config['code_length']+config['data_flow_length']-len(source_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        source_ids+=[tokenizer.pad_token_id]*padding_length      
        
        
        reverse_index={}
        for idx,x in enumerate(dfg):  
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]
        
        
        return (source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg)


    
class VocabModel_ggnn_without_seq(object):
    """for ggnn"""
    def __init__(self,config=myConfig.get_config(),vocab_restore=False,canditate_restore=False,logger=None) -> None:        
        
        
        self.canditate_restore=canditate_restore
        self.vocab_restore=vocab_restore
        self.logger=logger
        self.vocab_size=config['ggnn']['vocab_size']
        self.min_word_freq=config['ggnn']['min_word_freq']
        self.config=config
        self.parser=CParser().parse
        self.reserved_tokens=["<pad>","<unk>","<s>","</s>"]
        
        self.word_embed_dim=config['ggnn']['word_embed_dim']
        self.build_vocab()
        self.PAD=self.trnas_node("<pad>")
        self.UNK=self.trnas_node("<unk>")
        self.num_edge_types=len(self.edge_idx2token)
        self.word_embeddings=self.randomize_embeddings()
       
        self.forbidden_tokens=forbidden_tokens
        self.build_all_candidate_tokens()
        self.all_candidate_idxs=self.tokenize_tokens2idxs(self.all_candidate_tokens)
        self.reserved_tokens_idxs=self.tokenize_tokens2idxs(self.reserved_tokens)
        
        self.random_attack=[insert.split(" ") for insert in inserts]
    def build_vocab(self):
        
        
        if os.path.isfile(self.config['vocab_set'].get('ggnn_load_path')) and not self.vocab_restore:
            vocab=read_pkl(self.config['vocab_set']['ggnn_load_path'])
            self.node_vocab_cnt=vocab['node_vocab_cnt']
            self.edge_vocab_cnt=vocab['edge_vocab_cnt']
            self.node_token2idx=vocab['node_token2idx']
            self.node_idx2token=vocab['node_idx2token']
            self.edge_token2idx=vocab['edge_token2idx']
            self.edge_idx2token=vocab['edge_idx2token']
        else:
            if self.logger is not None:
                self.logger.info(f"building ggnn vocab from {self.config['raw_data_path']['train']}")
            # init
            self.node_idx2token=self.reserved_tokens[:]
            self.node_token2idx=dict(zip(self.reserved_tokens, range(len(self.reserved_tokens))))
    
            self.edge_idx2token=["<pad>"]
            self.edge_token2idx=dict(zip(self.edge_idx2token, range(len(self.edge_idx2token))))

            self.node_vocab_cnt = Counter()
            self.edge_vocab_cnt = Counter()
            train_raw=read_pkl(self.config['raw_data_path']['train'])
            self.build_vocab_from_codes(train_raw['code'])
            vocab={'node_vocab_cnt':self.node_vocab_cnt,'node_token2idx':self.node_token2idx,'node_idx2token':self.node_idx2token,
                    'edge_vocab_cnt':self.edge_vocab_cnt,'edge_token2idx':self.edge_token2idx,'edge_idx2token':self.edge_idx2token}
            save_pkl(vocab,self.config['vocab_set']['ggnn_load_path'])
            if self.logger is not None:
                self.logger.info(f"saving vocab in {self.config['vocab_set']['ggnn_load_path']}")
    def build_vocab_from_codes(self,codes):
        
        for code in tqdm(codes):
            code_graph=self.code_to_graph(code)
            
            for node in code_graph['nodes']:
                token=self.getToken(node['content'])
                self.node_vocab_cnt.update([token])
            for edge in code_graph['edges']:
               self.edge_vocab_cnt.update([edge[0]])

        self._add_words(self.node_vocab_cnt.keys(),self.node_token2idx,self.node_idx2token)
        self._add_words(self.edge_vocab_cnt.keys(),self.edge_token2idx,self.edge_idx2token)
        self._trim(self.node_token2idx,self.node_idx2token,self.node_vocab_cnt,self.vocab_size,self.min_word_freq)
        self._trim(self.edge_token2idx,self.edge_idx2token,self.edge_vocab_cnt)
        self.vocab_size=len(self.node_idx2token)
    def _add_words(self,words,token2idx:dict,idx2token:list):
        
        for token in words:
            if token not in token2idx.keys():
                token2idx[token]=len(idx2token)
                idx2token.append(token)
    def _trim(self,token2idx:dict,idx2token:list,vocab_cnt:dict,vocab_size=5000,min_freq=1):
       
        if min_freq <= 1 and (vocab_size is None or vocab_size >= len(token2idx)):
            return
        ordered_words = sorted(((c, w) for (w, c) in vocab_cnt.items()), reverse=True)
        if vocab_size:
            ordered_words = ordered_words[:vocab_size-len(self.reserved_tokens)]
        idx2token = self.reserved_tokens[:]
        token2idx = dict(zip(self.reserved_tokens, range(len(self.reserved_tokens))))
        vocab_cnt = Counter()
        for count, token in ordered_words:
            if count < min_freq: break
            if token not in token2idx:
                token2idx[token] = len(idx2token)
                vocab_cnt[token] = count
                idx2token.append(token)
        assert len(idx2token) == len(token2idx)    
    def trnas_node(self, item):
       
        if type(item) in [int, numpy.int64]:
            if item<self.vocab_size:
                return self.node_idx2token[item]
            else:
                return "<unk>"
        else:
            item=self.getToken(item)
            if item in self.node_token2idx.keys():
                return self.node_token2idx[item]
            else:
                return self.node_token2idx["<unk>"]
    def tokenize_tokens2idxs(self,tokens):
        
        if isinstance(tokens,str):
            tokens=self.getCodeTokens(tokens)
        idxs=[]
        for token in tokens:
            token_list=self.subtokenizer(token)
            for t in token_list:
                idxs.append(self.trnas_node(t))   
        return idxs 
    def tokenize_idxs2tokens(self,idxs):
        
        tokens=[]
        for idx in idxs:
            tokens.append(self.trnas_node(idx))   
        return tokens 
    def trnas_edge(self, item):
        
        if type(item) in [int, numpy.int64]:
            if item<self.vocab_size:
                return self.edge_idx2token[item]
            else:
                return "<unk>"
        else:
            item=self.getToken(item)
            if item in self.edge_token2idx.keys():
                return self.edge_token2idx[item]
            else:
                return self.edge_token2idx["<unk>"]
    def randomize_embeddings(self,scale=0.08):
        
        vocab_size = self.get_vocab_size()
        shape = (vocab_size, self.word_embed_dim)
        self.embeddings = np.array(np.random.uniform(low=-scale, high=scale, size=shape), dtype=np.float32)
        self.embeddings[self.node_token2idx['<pad>']] = np.zeros(self.word_embed_dim)
    def get_vocab_size(self):
        self.vocab_size=len(self.node_idx2token)
        return self.vocab_size
    def load_embedding(self,file_path, scale=0.08, dtype=np.float32):
        
        hit_words = set()
        vocab_size = len(self)
        with open(file_path, 'rb') as f:
            for line in f:
                line = line.split()
                word = line[0].decode('utf-8')
                idx = self.node_token2idx.get(word.lower(), None)
                if idx is None or idx in hit_words:
                    continue

                vec = np.array(line[1:], dtype=dtype)
                if self.embeddings is None:
                    
                    n_dims = len(vec)
                    self.embeddings = np.array(np.random.uniform(low=-scale, high=scale, size=(vocab_size, n_dims)), dtype=dtype)
                    self.embeddings[self.node_token2idx['pad']] = np.zeros(n_dims)
                self.embeddings[idx] = vec
                hit_words.add(idx)
        print('Pretrained word embeddings hit ratio: {}'.format(len(hit_words) / len(self.index2word)))
    def code_to_graph(self,code):
        
        
        visitor=CAstGraphGenerator()
        visitor.visit(self.parser(code))
        edge_list = [(t, origin, destination)
                    for (origin, destination), edges
                    in visitor.graph.items() for t in edges]
        
        graph_node_labels = [str(label).strip() for (_, label) in sorted(visitor.node_label.items())]
        graph = {"edges": edge_list, "backbone_sequence": visitor.terminal_path, "node_labels": graph_node_labels}
        nor_graph=self.normalize_graph(graph)
        code_graph=self.filtered_graph(nor_graph)
        return code_graph
    
    def normalize_graph(self,graph):
       
        node_labels = graph['node_labels']
        backbone_sequence = graph['backbone_sequence']
        formatted_node_labels = []
        for index, node_label in enumerate(node_labels):
            formatted_node_labels.append({'id': index, 'contents': node_label, 'type': node_label})
        index = 0
        mapping = {}
        nodes_to_subtokenize = {}
        token_sequential_list = []
        seq_token_in_node = []
        method_nodes = []
        method_edges = []
        for i, sorted_node in enumerate(formatted_node_labels):
            if i in backbone_sequence:
                subtokens = self.subtokenizer(sorted_node['contents'])
                if len(subtokens) > 1:
                    for subtoken in subtokens:
                        dummy_node = sorted_node.copy()
                        dummy_node['contents'] = subtoken
                        dummy_node['id_sorted'] = index
                        dummy_node['subtoken'] = True
                        dummy_node['ori_token'] = sorted_node['contents']
                        token_sequential_list.append(subtoken)
                        seq_token_in_node.append(index)
                        method_nodes.append(dummy_node)
                        if sorted_node['id'] not in mapping.keys():
                            mapping[sorted_node['id']] = index
                        if sorted_node['id'] not in nodes_to_subtokenize.keys():
                            nodes_to_subtokenize[sorted_node['id']] = [index]
                        else:
                            nodes_to_subtokenize[sorted_node['id']].append(index)
                        index += 1
                else:
                    sorted_node['id_sorted'] = index
                    sorted_node['subtoken'] = False
                    method_nodes.append(sorted_node)
                    mapping[sorted_node['id']] = index
                    seq_token_in_node.append(index)
                    token_sequential_list.append(sorted_node['contents'])
                    index += 1
            else:
                sorted_node['id_sorted'] = index
                method_nodes.append(sorted_node)
                mapping[sorted_node['id']] = index
                index += 1
        edge_label_dict = {'child': 'AST_CHILD', 'NextToken': 'NEXT_TOKEN', 'computed_from': 'COMPUTED_FROM',
                        'last_use': 'LAST_USE', 'last_write': 'LAST_WRITE'}
        for edge in graph['edges']:
            if edge[0] in edge_label_dict.keys():
                method_edges.append({'type': edge_label_dict[edge[0]], 'sourceId': mapping[edge[1]], 'destinationId': mapping[edge[2]]})
        for key in nodes_to_subtokenize.keys():
            for index in range(1, len(nodes_to_subtokenize[key])):
                edge = {}
                edge['type'] = 'SUB_TOKEN'
                edge['sourceId'] = nodes_to_subtokenize[key][index-1]
                edge['destinationId'] = nodes_to_subtokenize[key][index]
                method_edges.append(edge)
        if not len(seq_token_in_node) == len(token_sequential_list):
            code_graph = {}
        else:
            code_graph = {'nodes': method_nodes, 'edges': method_edges, 'seq_token_in_node': seq_token_in_node,
                        'tokens': token_sequential_list}
        return code_graph
    
    def subtokenizer(self, identifier):
        
        splitter_regex = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        identifiers = re.split('[._\-]', identifier)
        subtoken_list = []
        for identifier in identifiers:
            matches = splitter_regex.finditer(identifier)
            for subtoken in [m.group(0) for m in matches]:
                subtoken_list.append(subtoken)
        return subtoken_list
    def filtered_graph(self,normalize_graph,isLower=False):
        backbone_sequence = normalize_graph['tokens']
        seq_token_in_node = normalize_graph['seq_token_in_node']
        if isLower:
            backbone_sequence = [token.lower() for token in backbone_sequence]
        
        filter_code_nodes = []
        edges = []
        node_id_list = []
        for node in normalize_graph['nodes']:
            if isLower:
                node_content = node['contents'].lower()
            else:
                node_content = node['contents']
            node_id = node['id_sorted']
            filter_code_nodes.append({'id': node_id, 'content': node_content, 'type': node['type']})
            node_id_list.append(node_id)
        for edge in normalize_graph['edges']:
            if edge['sourceId'] in node_id_list and edge['destinationId'] in node_id_list:
                edges.append([edge['type'], edge['sourceId'], edge['destinationId']])
       
        code_graph = {'nodes': filter_code_nodes, 'edges': edges}
        return code_graph   
    def __len__(self):
        
        return self.vocab_size

    def build_all_candidate_tokens(self):
        
        if os.path.isfile(self.config['node'].get('all_candidate_ggnn_tokens')) and not self.canditate_restore:
            self.all_candidate_tokens=read_pkl(self.config['node']['all_candidate_ggnn_tokens'])
        else:
            if self.logger is not None:
                self.logger.info(f"building all candidate tokens from {self.config['raw_data_path']['train']} and ggnn vocab")        
            old_candidate_tokens=self._build_all_candidate_tokens()
            
            candidate_tokens=[]
            
            for subtoken_idx in tqdm(range(self.vocab_size)):
                subtoken = self.trnas_node(subtoken_idx)
                assert isinstance(subtoken, str)
                
                if subtoken in self.reserved_tokens:
                    continue
                if self.getToken(subtoken) in ['char','<int>','<fp>','str']:
                    continue
            
                
                for token in old_candidate_tokens:
                    
                    if subtoken in self.subtokenizer(token) and \
                    subtoken not in candidate_tokens and \
                    subtoken not in self.forbidden_tokens:
                        candidate_tokens.append(subtoken)
                        break
                
    
            self.all_candidate_tokens=candidate_tokens
            save_pkl(self.all_candidate_tokens,self.config['node']['all_candidate_ggnn_tokens'])
    def _build_all_candidate_tokens(self):
        
        if os.path.isfile(self.config['node'].get('all_candidate_tokens')) :
            tokens=read_pkl(self.config['node']['all_candidate_tokens'])
            return tokens
        train_raw=read_pkl(self.config['raw_data_path']['train'])
        codes=train_raw['code']
        tokens=set()
        for code in tqdm(codes):
            tokens.update(VocabModel.get_code_candidate_tokens(code,None))
        
        tokens=tokens-set(forbidden_tokens)
        tokens=list(tokens)
        save_pkl(tokens,self.config['node']['all_candidate_tokens'])
        return list(tokens)
    def getToken(self,token):
        
        if token[0] == '"' and token[-1] == '"':
                return "<str>"
        elif token[0] == "'" and token[-1] == "'":
            return "<char>"
        elif token[0] in "0123456789.":
            if 'e' in token.lower():
                return "<fp>" 
            elif '.' in token:
                if token == '.':
                    return token
                else:
                    return "<fp>"
            else:
                return "<int>"
        else:
            return token        
    @staticmethod
    def getCodeTokens(code):        
        
        tokens=[]
        parser = pycparser.CParser()
        parser.clex.input(code)
        t = parser.clex.token()
        while t is not None:
            tokens.append(t.value)
            t = parser.clex.token()
        return tokens

    

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                        default='lstm', help="vocab name, choose from [lstm,gru,ggnn]")
    parser.add_argument('--canditate_restore', type=bool,
                        default=False, help="Whether to regenerate the dictionary")
    args = parser.parse_args()
    vocab_name = args.name
    canditate_restore=args.canditate_restore
    vocab_name = args.name
    set_seed()
    config=myConfig().config
    vocab_map = {
             "lstm": VocabModel, 
             'gru': VocabModel, 
             'codebert': VocabModel_codebert, 
             "graphcodebert": VocabModel_graphcodebert, 
             'ggnn':VocabModel_ggnn_without_seq}
    vocab=vocab_map[vocab_name](config=config ,vocab_restore=False,canditate_restore=False)
    

