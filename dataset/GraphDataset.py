'''
Author: Zhang
Date: 2022-08-12
LastEditTime: 2023-01-22
FilePath: /code_attack/csa/dataset/GraphDataset.py


'''
import sys
sys.path.append("..")
import os
from utils.tools import *
from utils.basic_setting import *
from vocabClass import VocabModel_ggnn_without_seq
from CodeFeature import CodeFeatures_ggnn
from random import shuffle,sample
from tqdm import tqdm
from tqdm.contrib import tzip
class Batch_s(object):
    def __init__(self, instances, vocab:VocabModel_ggnn_without_seq):
        self.instances = instances
        self.batch_size = len(instances)
        # Create word representation and length
        
        self.code_indexs=[]
        self.labels=[]
        
    

        batch_code_graph = []
        for codefeature in instances:
            self.code_indexs.append(codefeature.id)
            batch_code_graph.append(codefeature.graph)
            self.labels.append(codefeature.label)
           
        batch_code_graphs = GGNNDataset_s.cons_batch_graph(batch_code_graph, vocab)
        self.code_graph =  GGNNDataset_s.vectorize_batch_graph(batch_code_graphs, vocab)

from scipy.sparse import *
class GGNNDataset_s(object):
    def __init__(self, vocab,type='train',model_name='ggnn',restore=False,batches_restore=False,isShuffle=True, isLoop=True, isSort=False, batch_size=-1,config=None,is_enhance=False,enhance_type=None,sample_nums=None):
        
        self.vocab=vocab
        self.type=type 
        self.model_name=model_name
        self.restore=restore
        self.batches_restore=batches_restore
        self.config=config
        if not is_enhance :
            self.data=self.load_data()
        else:
            if self.config[model_name]['graph_type']!='ggnn_bi':
                model_enhance_path=self.config[model_name]["enhance_path"].replace('ggnn','gcn')
            else: model_enhance_path=self.config[model_name]["enhance_path"]
            
            if sample_nums is not None and sample_nums>0:
                enhance_path=os.path.join(model_enhance_path,f"{type}_{enhance_type}_{str(sample_nums)}.pkl")
                adv_datas_path=os.path.join(model_enhance_path,f"{type}_{enhance_type}_retrain_{str(sample_nums)}_{config[model_name]['enhance_size']}.pkl")
            else:
                enhance_path=os.path.join(model_enhance_path,f"{type}_{enhance_type}.pkl")
                adv_datas_path=os.path.join(model_enhance_path,f"{type}_{enhance_type}_retrain_{config[model_name]['enhance_size']}.pkl")
            self.data=self.load_adv_date(adv_datas_path,enhance_path,enhance_size=self.config[model_name]["enhance_size"])
            self.config[self.model_name]['batches'][self.type]=self.config[self.model_name]['batches'][self.type][:-4]+f"_adv{self.config[self.model_name]['enhance_size']}.pkl"
        self.data_nums=len(self.data)
        self.batch_size=batch_size if batch_size!=-1 else config['ggnn']['batch_size']
        self.isSort=isSort
        self.isShuffle = isShuffle
        self.isLoop = isLoop
        self.cur_pointer = 0
        self.init_batches()
            
    def init_batches(self):
        if os.path.isfile(self.config[self.model_name]['batches'][self.type]) and not self.restore and not self.batches_restore:
            self.batches=read_pkl(self.config[self.model_name]['batches'][self.type])
        else:
            if self.isSort:
                self.data = sorted(self.data, key=lambda instance: (instance[1].get_node_length()))
            else:
                shuffle(self.data)
            
            # # distribute srcs into different buckets
            batch_spans = self.make_batches(self.data_nums,  self.batch_size)
            self.batches = []
            for (batch_start, batch_end) in tqdm(batch_spans):
                cur_instances = self.data[batch_start: batch_end]
                cur_batch = Batch_s(cur_instances, self.vocab)
                self.batches.append(cur_batch)
            save_pkl(self.batches,self.config[self.model_name]['batches'][self.type])
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        if self.isShuffle: np.random.shuffle(self.index_array)
    
    @staticmethod
    def get_model_input(codefeatures_list,vocab):
        if not isinstance(codefeatures_list,list):
            codefeatures_list=[codefeatures_list]
        cur_batch=Batch_s(codefeatures_list, vocab)
        return cur_batch
    def nextBatch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        
        return cur_batch

    def resetBatch(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]
    def load_data(self):
        
        if os.path.isfile(self.config[self.model_name][self.type]) and not self.restore:
            return read_pkl(self.config[self.model_name][self.type])
        else:
            
            with gzip.open(self.config['raw_data_path'][self.type]) as f:
                raw_data=pickle.load(f)
            ids=raw_data['id']
            codes=raw_data['code']
            labels=raw_data['label']
            
            features = [CodeFeatures_ggnn(code_id,code,label,self.vocab) for code_id,code,label in tzip(ids,codes,labels)]
            save_pkl(features,self.config[self.model_name][self.type])
            return features
    def load_adv_date(self,adv_datas_path,enhance_path,enhance_size):
        if os.path.isfile(adv_datas_path) and not self.restore:
            return read_pkl(adv_datas_path)
        else:
            
            adv_datas=self.load_data()
            enhance_date=read_pkl(enhance_path)
            if enhance_size == 'None':
                enhance_size=len(enhance_date)
            else:
                enhance_size=min(int(enhance_size),len(enhance_date))
            add_datas=sample(enhance_date,enhance_size)
            adv_datas+=add_datas
            adv_datas=sample(adv_datas,len(adv_datas))# 打乱顺序
            save_pkl(adv_datas,adv_datas_path)
            return adv_datas
    def __len__(self):
        return self.data_nums
    @staticmethod
    def make_batches(size, batch_size):   
        
        nb_batch = int(np.ceil(size/float(batch_size)))
        return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]
    @staticmethod
    def cons_batch_graph(graphs, vocab):
        
        num_nodes = max([len(g['nodes']) for g in graphs])
        num_edges = max([len(g['edges']) for g in graphs])
        batch_edges = []
        batch_node2edge = []
        batch_edge2node = []
        batch_node_num = []
        batch_node_index = []
        for g in graphs:
            edges = {}
            graph_node_index = GGNNDataset_s.cons_node_features(g['nodes'], vocab)
            node2edge = lil_matrix(np.zeros((num_edges, num_nodes)), dtype=np.float32)
            edge2node = lil_matrix(np.zeros((num_nodes, num_edges)), dtype=np.float32)
            edge_index = 0
            for edge, src_node, dest_node in g['edges']:
                if src_node == dest_node:  # Ignore self-loops for now
                    continue
                edges[edge_index] = edge
                node2edge[edge_index, dest_node] = 1# edge into node
                edge2node[src_node, edge_index] = 1 # node into edge
                edge_index += 1
            batch_edges.append(edges)
            batch_node2edge.append(node2edge)
            batch_edge2node.append(edge2node)
            batch_node_num.append(len(g['nodes']))
            batch_node_index.append(graph_node_index)
        batch_graphs = {'max_num_edges': num_edges,
                        'edge_features': batch_edges,
                        'node2edge': batch_node2edge,
                        'edge2node': batch_edge2node,
                        'node_num': batch_node_num,
                        'max_num_nodes': num_nodes,
                        'node_word_index': batch_node_index
                        }
        return batch_graphs
    
    @staticmethod
    def cons_node_features(nodes, vocab):
        
        
        graph_node_index = []
        for node in nodes:
            idx = vocab.trnas_node(node['content'])
            graph_node_index.append(idx)
        return graph_node_index

    @staticmethod
    def vectorize_batch_graph(graph, vocab):
        # 
        edge_features = []
        for edges in graph['edge_features']:
            edges_v = []
            for idx in range(len(edges)):
                edges_v.append(vocab.trnas_edge(edges[idx]))
            for _ in range(graph['max_num_edges'] - len(edges_v)):
                edges_v.append(vocab.trnas_edge("<pad>"))
            edge_features.append(edges_v)
            # z
        gv = {'edge_features': np.array(edge_features),
            'node2edge': graph['node2edge'],
            'edge2node': graph['edge2node'],
            'node_num': graph['node_num'],
            'max_node_num_batch': graph['max_num_nodes'],
            'node_index': GGNNDataset_s.pad_2d_vals_no_size(graph['node_word_index']) 
            }
        return gv
    @staticmethod
    def  vectorize_input(batch, training=True, device=None,is_first=True, mode='train'): 
        if not batch:
            return None
        if is_first:
            batch.labels=torch.LongTensor(batch.labels).to(device)
            
            batch.code_graph['node_index'] = torch.LongTensor(batch.code_graph['node_index']).to(device)
            batch.code_graph['node_num'] = torch.LongTensor(batch.code_graph['node_num'])# 
            batch.code_graph['edge_features']=torch.LongTensor(batch.code_graph['edge_features']).to(device)
            if mode=='grad':
                training=True
            
        with torch.set_grad_enabled(training):
            example = {'batch_size': batch.batch_size,
                    'code_graphs': batch.code_graph,                   
                    'labels':batch.labels
                    }
            return example
    @staticmethod
    def pad_2d_vals_no_size(in_vals, dtype=np.int32):
        size1 = len(in_vals)
        size2 = np.max([len(x) for x in in_vals])
        return GGNNDataset_s.pad_2d_vals(in_vals, size1, size2, dtype=dtype)

    @staticmethod
    def pad_2d_vals(in_vals, dim1_size, dim2_size, dtype=np.int32):
        out_val = np.zeros((dim1_size, dim2_size), dtype=dtype)
        if dim1_size > len(in_vals): dim1_size = len(in_vals)
        for i in range(dim1_size):
            cur_in_vals = in_vals[i]
            cur_dim2_size = dim2_size
            if cur_dim2_size > len(cur_in_vals): cur_dim2_size = len(cur_in_vals)
            out_val[i,:cur_dim2_size] = cur_in_vals[:cur_dim2_size]
        return out_val


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                        default='ggnn', help="dataset name, choose from [lstm,gru,codebert,graphcodebert]")
    parser.add_argument('--is_restore', type=bool,
                        default=False, help="Whether to regenerate the dataclass")
    args = parser.parse_args()
    restore=args.is_restore
    model_name=args.name
    config=myConfig().config
    n_gpu,device=set_device(config)
    config['device']=device
    vocab=VocabModel_ggnn_without_seq(config=config)
    restore=True
    batches_restore=True
    train=GGNNDataset_s(vocab,'train',restore=restore,batches_restore=batches_restore,config=config)
    dev=GGNNDataset_s(vocab,'dev',restore=restore,batches_restore=batches_restore,config=config)
    test=GGNNDataset_s(vocab,'test',restore=restore,batches_restore=batches_restore,config=config)
    print()
  
