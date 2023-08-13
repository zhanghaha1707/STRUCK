'''
Author: Zhang haha
Date: 2022-08-04 10:05:04
LastEditTime: 2023-01-22
Description: Record the relationship between supported files and mappings
'''
import sys
from dataset.CodeFeature import CodeFeatures, CodeFeatures_astnn, CodeFeatures_codebert, CodeFeatures_ggnn, CodeFeatures_graphcodebert, CodeFeatures_tbcnn
sys.path.append("..")
from dataset.dataClass import LSTMDataset, TBCNNDataset, ASTNNDataset, CodeBertDataset, GraphCodeBertDataset
from dataset.vocabClass import VocabModel, VocabModel_tbcnn, VocabModel_astnn, VocabModel_codebert, VocabModel_graphcodebert,VocabModel_ggnn_without_seq
from dataset.GraphDataset import GGNNDataset_s
from torch import optim
from model.GraphClassify_s import GraphClassifier as GraphClassifier_s
from model.GraphCodeBert import GraphCodeBERTClassifier
from model.CodeBert import CodeBERTClassifier
from model.GRU import GRUClassifier
from model.LSTM import LSTMClassifier
import torch.nn as nn

codefeatures_map={
            "lstm": CodeFeatures, 
             'gru': CodeFeatures, 
             'tbcnn': CodeFeatures_tbcnn, 
             'astnn': CodeFeatures_astnn,
             'codebert': CodeFeatures_codebert, 
             "graphcodebert": CodeFeatures_graphcodebert, 
             'ggnn': CodeFeatures_ggnn,
             'ggnn_simple':CodeFeatures_ggnn
}

vocab_map = {
             "lstm": VocabModel, 
             'gru': VocabModel, 
             'codebert': VocabModel_codebert, 
             "graphcodebert": VocabModel_graphcodebert, 
             'ggnn':VocabModel_ggnn_without_seq,
             'ggnn_simple':VocabModel_ggnn_without_seq}
support_dataset = {
                    "lstm": LSTMDataset, 
                    "gru": LSTMDataset, 
                    'codebert': CodeBertDataset, 
                    'graphcodebert': GraphCodeBertDataset, 
                    'ggnn':GGNNDataset_s,
                    'ggnn_simple':GGNNDataset_s}
support_model = {
                "lstm": LSTMClassifier, 
                 'gru': GRUClassifier, 
                 'codebert': CodeBERTClassifier, 
                 'graphcodebert': GraphCodeBERTClassifier, 
                 
                 'ggnn':GraphClassifier_s,
                 'ggnn_simple':GraphClassifier_s
                 }
optimizer_map = {
                'Adam': optim.Adam, 
                "SGD": optim.SGD,
                 "Adamax": optim.Adamax,
                 "AdamW": optim.AdamW
                 }
lossFunc_map = {"CrossEntropy": nn.CrossEntropyLoss}
support_attack = {"grad_rename": None,
                  "random_rename": None, 
                  "random_insert": None}
# 相关基础设置的config的映射
attack_map = {"grad_rename": "node", 
              "random_rename": "node",
              "mhm": "node_mhm",
              "pretrain":"node_pretrain",
              "insert": "struct",
              "random_insert":"random_struct",
              "insert_copy_decl": "struct",
              "copy_insert":"struct",
              "decl_insert":"struct",
              "redundant":"struct",
              "new_struct":"new_struct"
              }

