import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import sys
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.utils import train_test_split_edges_clga
import statistics
device = "cuda" if torch.cuda.is_available() else "cpu"
class HashAgent():
    def __init__(self,h="md5",T=30, p=0.5):
        '''
            h: the hash function in "md5","sha1","sha256"
            T: the subset amount
        '''

        super(HashAgent, self).__init__()
        self.T = T
        self.h= h 
        self.add_I = [ [] for _ in range(self.T)]
        
        for i in range(self.T):
            for j in range(self.T):
                if j==i:
                    continue
                if np.random.random()<=p:
                    self.add_I[i].append(j)
                                         
    def hash_edge(self,V, u,v):
        #"""
        hexstring = hex(V*u+v)
        hexstring= hexstring.encode()
        if self.h == "md5":
            hash_device = hashlib.md5()
        elif self.h == "sha1":
            hash_device = hashlib.sha1()
        elif self.h == "sha256":
            hash_device = hashlib.sha256()
        hash_device.update(hexstring)
        I = int(hash_device.hexdigest(),16)%self.T
        return I
    
    def generate_mixed_subgraphs(self, graph):
        
        mixed_subgraphs = []
        
        original = graph.edge_index
        nodes = range(graph.x.shape[0])

        V= graph.x.shape[0]
        
        
                    
        for i in range(self.T):
            mixed_subgraphs.append(Data(
                        x = graph.x,
                        y = graph.y,
                        edge_attr = graph.edge_attr,
                        edge_index = []
                    ))
            
        for i in tqdm(range(len(original[0])), desc="[generate_mixed_subgraphs] Original Edges"):
            
            u=original[0,i]
            v=original[1,i]
            if u>v:
                I = self.hash_edge(V,v,u)
            else:
                I = self.hash_edge(V,u,v)
            mixed_subgraphs[I].edge_index.append([u,v])
            
        for i in tqdm(range(V-1), desc="[generate_mixed_subgraphs] Hybrid Edges"):
            for j in range(i+1,V):
                u=nodes[i]
                v=nodes[j]
                I = self.hash_edge(V,u,v)
                for k in range(len(self.add_I[I])):
                    if self.add_I[I][k]==I:
                        continue
                    mixed_subgraphs[self.add_I[I][k]].edge_index.append([u,v])
                    mixed_subgraphs[self.add_I[I][k]].edge_index.append([v,u])
                    
        deletes = []
        new_mixed_subgraphs = []
        for i in range(self.T):
            if len(mixed_subgraphs[i].edge_index)==0:
                continue
            mixed_subgraphs[i].edge_index = torch.tensor(mixed_subgraphs[i].edge_index,dtype=torch.int64).transpose(1,0)
            new_mixed_subgraph = train_test_split_edges_clga(mixed_subgraphs[i])
            new_mixed_subgraphs.append(new_mixed_subgraph)
            
        return new_mixed_subgraphs#mixed_subgraphs


def enlarge_graph(dataset,hasher):
    new_graphs = []
    grounds = []
    
    times=0
    stds=[]
    avg_e = []
    min_e =[]
    max_e =[]
    train_index= []
    val_index= []
    test_index= []
    for i in range(len(dataset.graphs)):
        start = len(new_graphs)
        graph = dataset.graphs[i]
        n = graph.x.shape[0]
        nodes = [v for v in range(n)]
        ground = torch.zeros((n,n)).to(dataset.device)
        for j in range(graph.edge_index[0].shape[0]):
            ground[graph.edge_index[0,j],graph.edge_index[1,j]]=1
        explanation = dataset.explanations[i]
        new_graphs.append(graph)
        grounds.append(ground)
        mixed_graphs=hasher.generate_mixed_subgraphs(graph)
        
        if i %100 ==0:
            print(f"{i}/{len(dataset.graphs)}")
            
        edges = []
        new_graphs.extend(mixed_graphs)
        
        end = len(new_graphs)
        indexs = [j for j in range(start,end)]
        if i in dataset.train_index:
            train_index.extend(indexs)
        elif i in dataset.val_index:
            val_index.extend(indexs)
        elif i in dataset.test_index:
            test_index.extend(indexs)
            
        grounds.extend([ground for _ in range(len(mixed_graphs))])
        
    dataset.graphs = new_graphs
    dataset.ground = grounds
    
    dataset.train_index = torch.tensor(train_index)
    dataset.val_index = torch.tensor(val_index)
    dataset.test_index = torch.tensor(test_index)
    
    return  dataset
        
    
def enlarge_single_graph(graph,hasher, filename=None):
    new_graphs = []

    n = graph.x.shape[0]
    ground = torch.zeros((n,n)).to(graph.x.device)
    for j in range(graph.edge_index[0].shape[0]):
        ground[graph.edge_index[0,j],graph.edge_index[1,j]]=1
    new_graphs.append(graph)
    mixed_graphs=hasher.generate_mixed_subgraphs(graph)
    
    new_graphs.extend(mixed_graphs)
        
    dataset = Data()
    dataset.graphs = new_graphs

    if filename:
        torch.save(dataset, filename)
    return dataset
        
    
class RobustClassifier(nn.Module):
    def __init__(self,BaseClassifier,Hasher):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(RobustClassifier, self).__init__()
        self.BaseClassifier = BaseClassifier
        self.Hasher = Hasher
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
    def forward(self, graph):
        subgraphs = self.Hasher.generate_mixed_subgraphs(graph)
        
        for i in range(len(subgraphs)):
            subgraphs[i].exp_key = [i]
        loader = DataLoader(subgraphs, batch_size = len(subgraphs), shuffle = True)
        
        #x = torch.cat([subgraphs[i].x for i in range(len(subgraphs))]).to(self.device)
        #edge_index = torch.cat([subgraphs[i].edge_index for i in range(len(subgraphs))]).to(self.device)
        self.BaseClassifier.eval()
        outputs = []
        for i in range(len(subgraphs)):
            data=subgraphs[i]
            output = self.BaseClassifier(data.x,data.edge_index,batch =data.batch).cpu().detach()
            outputs.append(output)
        outputs = np.array(outputs)
        Y_labels = np.argmax(outputs,axis=1)
        vote_label = np.argmax(np.bincount(Y_labels))
        return vote_label
    def vote(self, graph):
        subgraphs = self.Hasher.generate_mixed_subgraphs(graph)
        
        for i in range(len(subgraphs)):
            subgraphs[i].exp_key = [i]
        loader = DataLoader(subgraphs, batch_size = len(subgraphs), shuffle = True)
        
        self.BaseClassifier.eval()
        outputs=[]
        Y_labels= []
        for i in range(len(subgraphs)):
            data=subgraphs[i].to(device)
            output = self.BaseClassifier(data.x,data.edge_index,batch =data.batch).cpu().detach()

            Y_labels.append(np.argmax(output,axis=1).item())
            
        count = np.bincount(Y_labels)
        votes = count.copy()
        vote_label = np.argmax(count)
        Yc = count[vote_label]
        count[vote_label]=-1
        second_label = np.argmax(count)
        Yb = count[second_label]
        
        if vote_label>second_label:
            Mc = (Yc-Yb-1)//2
        else:
            Mc = (Yc-Yb)//2
        return vote_label, Mc
    
