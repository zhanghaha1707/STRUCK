'''
Author: Zhang haha
Date: 2022-07-31 23:14:23
LastEditTime: 2023-01-20
Description: lstm
FilePath: /code_attack/csa/model/LSTM.py
'''
import torch
import torch.nn as nn
import numpy as np
import logging
logger=logging.getLogger(__name__)
class LSTMEncoder(nn.Module):
    
    def __init__(self,**config) -> None:
        super(LSTMEncoder, self).__init__()
        self.embedding_dim = config['embedding_size']
        self.hidden_dim = config['hidden_size']
        self.n_layers = config['n_layers']
        self.bidirectional = config['brnn']
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 
                            self.n_layers, dropout=config['encoder_dropout'], bidirectional=config['brnn'],batch_first=config['batch_first'])
    def forward(self, input, hidden=None):
       
        return self.lstm(input, hidden)
class LSTMClassifier(nn.Module):
   
    def __init__(self,**config):
        
        
        super(LSTMClassifier, self).__init__()
        
        self.vocab_size = config['vocab_size']
        self.embedding_size = config['embedding_size']
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size,padding_idx=0)# 建立embedding矩阵
        self.encoder = LSTMEncoder(**config)
        self.hidden_dim = config['hidden_size'] * 2 if self.encoder.bidirectional else config['hidden_size']# 最后获得表示的大小
        self.classify = nn.Linear(self.hidden_dim, config['num_classes'])
        self.Dropout = nn.Dropout(config['atten_dropout'])
        self.max_len = config['max_len']
       
        self.attn = config['attn']
        if self.attn:
            self.W = nn.Parameter(torch.Tensor(np.zeros((self.hidden_dim, 1))))#[hidden_dim,1]
        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))
        
    def forward(self, inputs,mask):
       
        self.encoder.lstm.flatten_parameters()
        emb = self.embedding(inputs)       
        outputs, hidden = self.encoder(emb)
        # attention
        if self.attn:
            M = nn.Tanh()(outputs)
            M = torch.reshape(M, [-1, self.hidden_dim])
            alpha = torch.mm(M, self.W)
            alpha = torch.reshape(alpha, [-1, self.max_len, 1])# batch
            alpha = nn.Softmax(dim=1)(alpha)
            A = outputs.permute([0, 2, 1])
            r = torch.bmm(A, alpha)
            r = torch.squeeze(r)    
            h_star = nn.Tanh()(r)
            drop = self.Dropout(h_star)
        else:
            masked=mask.unsqueeze(dim=-1)*outputs
            M = nn.Tanh()(masked)
            drop = (torch.max(outputs, dim=1)[0]).to(torch.float32) 
        
        logits = self.classify(drop)    # (B, H * direc)=>(B, Classes)
        return logits, emb
    
    def prob(self, inputs,mask):
       
        with torch.no_grad():
            logits = self.forward(inputs,mask)[0]
            prob = nn.Softmax(dim=1)(logits)# 概率
        return prob
        
    
    def grad(self, inputs,mask,labels, loss_fn):
        
        savep1 = self.encoder.lstm.dropout
        savep2 = self.Dropout.p
        
        self.encoder.lstm.dropout = 0
        self.Dropout.p = 0
        self.zero_grad()
        logits, emb = self.forward(inputs,mask)
        emb.retain_grad() 
        loss = loss_fn(logits, labels)
        loss.backward()
        
        self.encoder.lstm.dropout = savep1
        self.Dropout.p = savep2
        
        return emb.grad 