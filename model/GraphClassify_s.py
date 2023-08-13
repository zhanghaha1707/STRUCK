'''
Author: Zhang
Date: 2022-08-12
LastEditTime: 2023-01-20


'''
import sys
sys.path.append("..")
from dataset.vocabClass import VocabModel_ggnn_without_seq
sys.path.append("..")
from utils.tools import *
import torch
from torch import nn
import torch.nn.functional as F
from .GCN import GCN
import math, copy
INF = 1e20
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, config, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.config = config

    def forward(self, query, key, value, mask=None):
       
        # mask must be four dimension
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask,
                                      dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -INF)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        '''GatedFusion module'''
        self.fc_z = nn.Linear(4 * hidden_size, hidden_size, bias=True)

    def forward(self, h_state, input):
        z = torch.sigmoid(self.fc_z(torch.cat([h_state, input, h_state * input, h_state - input], -1)))
        h_state = (1 - z) * h_state + z * input
        return h_state


class GRUStep(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(GRUStep, self).__init__()
        '''GRU module'''
        self.linear_z = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_r = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_t = nn.Linear(hidden_size + input_size, hidden_size, bias=False)

    def forward(self, h_state, input):
        z = torch.sigmoid(self.linear_z(torch.cat([h_state, input], -1)))
        r = torch.sigmoid(self.linear_r(torch.cat([h_state, input], -1)))
        t = torch.tanh(self.linear_t(torch.cat([r * h_state, input], -1)))
        h_state = (1 - z) * h_state + z * t
        return h_state


def dropout(x, drop_prob, shared_axes=[], training=False):
    """
    Apply dropout to input tensor.
    Parameters
    ----------
    input_tensor: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)``
    Returns
    -------
    output: ``torch.FloatTensor``
        A tensor of shape ``(batch_size, ..., num_timesteps, embedding_dim)`` with dropout applied.
    """
    if drop_prob == 0 or drop_prob == None or (not training):# 调用dropout(node_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
        return x

    sz = list(x.size())
    for i in shared_axes:
        sz[i] = 1
    mask = x.new(*sz).bernoulli_(1. - drop_prob).div_(1. - drop_prob)
    mask = mask.expand_as(x)
    return x * mask



class  GraphMessagePassing(nn.Module):
    def __init__(self, config):
        super(GraphMessagePassing, self).__init__()
        self.config = config
        hidden_size = config['graph_hidden_size']
        if config['message_function'] == 'edge_mm':
            self.edge_weight_tensor = torch.Tensor(config['num_edge_types'], hidden_size * hidden_size)
            self.edge_weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.edge_weight_tensor))
            self.mp_func = self.msg_pass_edge_mm
        elif config['message_function'] == 'edge_network':
            self.edge_network = torch.Tensor(config['edge_embed_dim'], hidden_size, hidden_size)
            self.edge_network = nn.Parameter(nn.init.xavier_uniform_(self.edge_network))
            self.mp_func = self.msg_pass_edge_network
        elif config['message_function'] == 'edge_pair':
            self.linear_edge = nn.Linear(config['edge_embed_dim'], hidden_size, bias=False)
            self.mp_func = self.msg_pass
        elif config['message_function'] == 'no_edge':
            self.mp_func = self.msg_pass
        else:
            raise RuntimeError('Unknown message_function: {}'.format(config['message_function']))

    def msg_pass(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state)                      # batch_size x num_edges x hidden_size
        if edge_vec is not None and self.config['message_function'] == 'edge_pair':
            node2edge_emb = node2edge_emb + self.linear_edge(edge_vec)
        agg_state = torch.bmm(edge2node, node2edge_emb)                         # consider self-loop if preprocess not igore
        return agg_state

    def msg_pass_edge_mm(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size
        new_node2edge_emb = []
        for i in range(node2edge_emb.size(1)):
            edge_weight = F.embedding(edge_vec[:, i], self.edge_weight_tensor).view(-1, node_state.size(-1), node_state.size(-1)) # batch_size x hidden_size x hidden_size
            new_node2edge_emb.append(torch.matmul(edge_weight, node2edge_emb[:, i].unsqueeze(-1)).squeeze(-1))
        new_node2edge_emb = torch.stack(new_node2edge_emb, dim=1) # batch_size x num_edges x hidden_size
        agg_state = torch.bmm(edge2node, new_node2edge_emb)
        return agg_state

    def msg_pass_edge_network(self, node_state, edge_vec, node2edge, edge2node):
        node2edge_emb = torch.bmm(node2edge, node_state) # batch_size x num_edges x hidden_size
        new_node2edge_emb = []
        for i in range(node2edge_emb.size(1)):
            edge_weight = torch.mm(edge_vec[:, i], self.edge_network.view(self.edge_network.size(0), -1)).view((-1,) + self.edge_network.shape[-2:])
            new_node2edge_emb.append(torch.matmul(edge_weight, node2edge_emb[:, i].unsqueeze(-1)).squeeze(-1))
        new_node2edge_emb = torch.stack(new_node2edge_emb, dim=1) # batch_size x num_edges x hidden_size
        agg_state = torch.bmm(edge2node, new_node2edge_emb)
        return agg_state

class GraphNN(nn.Module):
    def __init__(self, config):
        super(GraphNN, self).__init__()
        print('[ Using {}-hop GraphNN ]'.format(config['graph_hops']))
        self.device = config['device']
        hidden_size = config['graph_hidden_size']
        self.hidden_size = hidden_size
        self.graph_direction = config['graph_direction']
        self.graph_type = config['graph_type']
        self.graph_hops = config['graph_hops']
        self.word_dropout = config['word_dropout']
        
        self.linear_max = nn.Linear(hidden_size, hidden_size, bias=False)
        if self.graph_type == 'ggnn_bi':
            self.static_graph_mp = GraphMessagePassing(config)
            self.static_gru_step = GRUStep(hidden_size, hidden_size)
            if self.graph_direction == 'all':
                self.static_gated_fusion = GatedFusion(hidden_size)
            self.graph_update = self.static_graph_update
        elif self.graph_type == 'gcn':
            self.gcn = GCN(config)
            self.graph_update = self.graph_gcn_update
        elif self.graph_type == 'gat':
            self.gat = GAT(config)
            self.graph_update = self.graph_gat_update

        print('[ Using graph type: {} ]'.format(self.graph_type))
        print('[ Using graph direction: {} ]'.format(self.graph_direction))

    def forward(self, node_feature, edge_vec, adj):
        node_state = self.graph_update(node_feature, edge_vec, adj)
        return node_state

    def static_graph_update(self, node_feature, edge_vec, adj):
        '''Static graph update'''
        node2edge, edge2node = adj
        # Shape: (batch_size, num_edges, num_nodes) x.A=><153x60 sparse matrix of type '<class 'numpy.float32'>'with 93 stored elements in List of Lists format>
        node2edge = torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0).to(self.device)
        # Shape: (batch_size, num_nodes, num_edges)
        edge2node = torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0).to(self.device)
        for _ in range(self.graph_hops):
            bw_agg_state = self.static_graph_mp.mp_func(node_feature, edge_vec, node2edge, edge2node)  # (num_nodes, dim) 
            fw_agg_state = self.static_graph_mp.mp_func(node_feature, edge_vec, edge2node.transpose(1, 2), node2edge.transpose(1, 2))
            if self.graph_direction == 'all':
                agg_state = self.static_gated_fusion(fw_agg_state, bw_agg_state)
                node_feature = self.static_gru_step(node_feature, agg_state)
            elif self.graph_direction == 'forward':
                node_feature = self.static_gru_step(node_feature, fw_agg_state)
            else:
                node_feature = self.static_gru_step(node_feature, bw_agg_state)
        return node_feature

    def graph_gat_update(self, node_state, edge_vec, adj, node_mask=None):
        node2edge, edge2node = adj
        node2edge = torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0).to(self.device)
        # Shape: (batch_size, num_nodes, num_edges)
        edge2node = torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0).to(self.device)
        adj = torch.bmm(edge2node, node2edge)
        node_states=[]
        for index in range(node_state.size(0)):

            current_node_state = node_state[index, :, :]
            current_adj = adj[index, :, :]
            gat_node_state = self.gat(current_node_state, current_adj)
            node_states.append(gat_node_state)
        node_states = torch.stack(node_states)
        return node_states

    def graph_gcn_update(self, node_state, edge_vec, adj, node_mask=None):
        # node_state是
        '''gcn graph update'''
        node2edge, edge2node = adj
        node2edge = torch.stack([torch.Tensor(x.A) for x in node2edge], dim=0).to(self.device)# 将list转换为tensor
        # Shape: (batch_size, num_nodes, num_edges)
        edge2node = torch.stack([torch.Tensor(x.A) for x in edge2node], dim=0).to(self.device)
        adj = torch.bmm(edge2node, node2edge)# batch的adj
        node_states=[]
        for index in range(node_state.size(0)):
            
            current_node_state = node_state[index, :, :]
            current_adj = adj[index, :, :]
            gcn_node_state = self.gcn(current_node_state, current_adj)
            node_states.append(gcn_node_state)
        node_states = torch.stack(node_states)
        return node_states

class GraphClassifier(nn.Module):

    def __init__(self,**config):
        super(GraphClassifier, self).__init__()
        self.name = 'GraphClassifier'
        self.vocab_size=config['vocab'].get_vocab_size()
        self.device = config['device']
        self.word_dropout = config['word_dropout']
        self.graph_hidden_size = config['graph_hidden_size']

        self.message_function = config['message_function']
        self.config=config
        self.vocab=config['vocab']
        if config['fix_word_embed']:
            print('[ Fix word embeddings ]')
            for param in self.word_embed.parameters():
                param.requires_grad = False
        
        self.word_embed=nn.Embedding(config['vocab_size'], config['word_embed_dim'], padding_idx=config['vocab'].PAD, _weight=torch.from_numpy(config['vocab'].word_embeddings).float()if config['vocab'].word_embeddings is not None else None)
        self.edge_embed = nn.Embedding(config['vocab'].num_edge_types, config['edge_embed_dim'], padding_idx=config['vocab'].PAD)
        self.code_graph_encoder = GraphNN(config)
        
        self.linear_max = nn.Linear(self.graph_hidden_size, self.graph_hidden_size, bias=False)
        self.heads = config.get('heads', 4)
        
        
        self.linear_proj = nn.Linear(self.graph_hidden_size, 2 * self.graph_hidden_size, bias=False)

        self.code_info_type = config['code_info_type']
        
        self.linear_classify=nn.Linear(self.graph_hidden_size,config['num_classes'])
        
        
    
    def forward(self, batch):
        code_graphs = batch['code_graphs']
        
        if self.message_function == 'edge_mm':
            code_edge_vec = code_graphs['edge_features']# batch*max_num_edges
        else:
            code_edge_vec = self.edge_embed(code_graphs['edge_features'])
          
        code_node_mask = create_mask(code_graphs['node_num'], code_graphs['max_node_num_batch']).to(self.device)
        

        node_embedded = self.word_embed(code_graphs['node_index'])
        node_embedded = dropout(node_embedded, self.word_dropout, shared_axes=[-2], training=self.training)
        

        
        code_node_embedding = self.code_graph_encoder(node_embedded, code_edge_vec,
                                                          (code_graphs['node2edge'], code_graphs['edge2node']))
        local_code_state = self.graph_maxpool(code_node_embedding, code_node_mask).squeeze(-1)
        
    
        
        logits=self.linear_classify(local_code_state)

       
        
        return logits
    def prob(self,batch):
        
        with torch.no_grad():
            logits = self.forward(batch)
            prob = nn.Softmax(dim=1)(logits)
        return prob
   
    def grad(self,batch,loss_function):
        '''
        Description: 获得测试的样例的梯度
        param batch [str]:测试集
        return [str] word_embedding层的梯度
        '''
        logits=self.forward(batch)
        labels = batch['labels']
        
        
        self.word_embed.weight.retain_grad()
        loss = loss_function(logits, labels)
        loss.backward()
        return self.word_embed.weight.grad    
    def softmax_loss(self, src_state, tgt_state, criterion, batch_size):
        logits = torch.matmul(tgt_state, src_state.transpose(0, 1))
        label = torch.arange(batch_size, dtype=torch.long).to(self.device)
        loss = criterion(logits, label)
        return loss, logits

    def graph_maxpool(self, node_state, node_mask=None):
        node_mask = node_mask.unsqueeze(-1)
        node_state = node_state * node_mask.float()
        node_embedding_p = self.linear_max(node_state).transpose(-1, -2)
        graph_embedding = F.max_pool1d(node_embedding_p, kernel_size=node_embedding_p.size(-1))
        return graph_embedding
