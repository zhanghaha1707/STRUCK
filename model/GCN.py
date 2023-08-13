import torch.nn as nn
import torch.nn.functional as F
import math	# 导入数学运算函数的模块

import torch
from torch.nn.parameter import Parameter	
from torch.nn.modules.module import Module	


class GraphConvolution(Module):	
	
	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		
		self.weight = Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias=Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter('bias', None)	
		self.reset_parameters()
	def reset_parameters(self):	
		stdv = 1. / math.sqrt(self.weight.size(1))
		
		self.weight.data.uniform_(-stdv, stdv)
		
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)	
	def forward(self, input, adj):
		support = torch.mm(input, self.weight)
		
		output = torch.spmm(adj, support)
       
		if self.bias is not None:	
			return output + self.bias	
		else:
			return output	
	def __repr__(self):	
		return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
		
	
class GCN(nn.Module):	
	def __init__(self, config):
		super(GCN, self).__init__()
		nfeat = config['word_embed_dim']
		nhid=config['graph_hidden_size']
		bias=config['bias']
		self.gc1 = GraphConvolution(nfeat, nhid,bias)
		
		self.gc2 = GraphConvolution(nhid, nhid,bias)
		self.dropout = config['gcn_dropout']
	def forward(self, x, adj):	
		x = F.relu(self.gc1(x, adj))
		x = F.dropout(x, self.dropout, training=self.training)
		
		x = self.gc2(x, adj)
		
		return x	
	

