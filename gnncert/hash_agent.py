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
from utils.utils import train_split_edges_clga, train_test_split_edges_clga
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
        V = graph.x.shape[0]
        
        # Create T empty subgraphs
        for i in range(self.T):
            mixed_subgraphs.append(Data(
                x = graph.x,
                y = graph.y,
                edge_attr = graph.edge_attr,
                edge_index = []
            ))
        
        # Only use training edges for subgraph creation
        train_edges = graph.train_pos_edge_index
        
        # First, distribute training edges to subgraphs
        for i in range(len(train_edges[0])):
            u = train_edges[0,i]
            v = train_edges[1,i]
            if u > v:
                I = self.hash_edge(V,v,u)
            else:
                I = self.hash_edge(V,u,v)
            mixed_subgraphs[I].edge_index.append([u,v])
            mixed_subgraphs[I].edge_index.append([v,u])
        
        # Then, generate hybrid edges only from training edges
        for i in tqdm(range(len(train_edges[0])), desc="[generate_mixed_subgraphs] Hybrid Edges"):
            u = train_edges[0,i]
            v = train_edges[1,i]
            I = self.hash_edge(V,u,v)
            for k in range(len(self.add_I[I])):
                if self.add_I[I][k]==I:
                    continue
                # Only add hybrid edge if it's not a val/test edge
                if not self.is_val_test_edge(u, v, graph):
                    mixed_subgraphs[self.add_I[I][k]].edge_index.append([u,v])
                    mixed_subgraphs[self.add_I[I][k]].edge_index.append([v,u])
        
        # Process subgraphs and split into train/val/test
        new_mixed_subgraphs = []
        for i in range(self.T):
            if len(mixed_subgraphs[i].edge_index)==0:
                continue
            mixed_subgraphs[i].edge_index = torch.tensor(mixed_subgraphs[i].edge_index,dtype=torch.int64).transpose(1,0)
            mixed_subgraphs[i].train_pos_edge_index =  mixed_subgraphs[i].edge_index
            new_mixed_subgraph = train_split_edges_clga(mixed_subgraphs[i])
            new_mixed_subgraph.num_nodes = graph.num_nodes
            new_mixed_subgraphs.append(new_mixed_subgraph)
            
        return new_mixed_subgraphs

    def is_val_test_edge(self, u, v, graph):
        # Check if edge (u,v) exists in validation or test sets
        val_edges = graph.val_pos_edge_index
        test_edges = graph.test_pos_edge_index
        
        # Check in validation edges
        val_mask = ((val_edges[0] == u) & (val_edges[1] == v)) | ((val_edges[0] == v) & (val_edges[1] == u))
        if val_mask.any():
            return True
        
        # Check in test edges
        test_mask = ((test_edges[0] == u) & (test_edges[1] == v)) | ((test_edges[0] == v) & (test_edges[1] == u))
        if test_mask.any():
            return True
        
        return False

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

    new_graphs.append(graph)
    mixed_graphs=hasher.generate_mixed_subgraphs(graph)
    
    new_graphs.extend(mixed_graphs)
        
    dataset = Data()
    dataset.graphs = new_graphs

    if filename:
        torch.save(dataset, filename)
    return dataset
        
    
class RobustEdgeClassifier(nn.Module):
    def __init__(self, BaseClassifier, Hasher):
        super(RobustEdgeClassifier, self).__init__()
        self.BaseClassifier = BaseClassifier
        self.Hasher = Hasher
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, graph, edge_index, subgraphs=None):
        # Generate subgraphs (only from training edges)
        if not subgraphs:
            subgraphs = self.Hasher.generate_mixed_subgraphs(graph)
        
        # Get predictions for each edge from each subgraph
        self.BaseClassifier.eval()
        edge_predictions = []  # List to store predictions for each subgraph
        
        for i, subgraph in enumerate(subgraphs):
            # Get predictions using the base classifier's forward method
            pred = self.BaseClassifier(subgraph.to(self.device), edge_index.to(self.device))
            edge_predictions.append(pred)
        
        # Stack predictions from all subgraphs
        edge_predictions = torch.stack(edge_predictions)
        # assert edge_predictions.shape[0] <= 2
        # if edge_predictions.shape[0] == 2:
        #     print(np.sum(edge_predictions[0].detach().cpu().numpy() != edge_predictions[1].detach().cpu().numpy()))
        #     print(len(edge_predictions[0]))
        # Aggregate predictions (mean across subgraphs)
        final_predictions = torch.mean(edge_predictions, dim=0)
                
        return final_predictions

    def vote(self, graph):
        # Generate subgraphs (only from training edges)
        subgraphs = self.Hasher.generate_mixed_subgraphs(graph)
        
        # Get predictions for each edge from each subgraph
        self.BaseClassifier.eval()
        edge_predictions = []  # List to store predictions for each subgraph
        
        for subgraph in subgraphs:
            # Get predictions using the base classifier's forward method
            pred = self.BaseClassifier(subgraph, graph.val_pos_edge_index)
            edge_predictions.append(pred)
        
        # Stack predictions from all subgraphs
        edge_predictions = torch.stack(edge_predictions)
        
        # For each edge, calculate margin between top predictions
        margins = torch.zeros(edge_predictions.shape[1])
        for i in range(edge_predictions.shape[1]):
            # Get predictions for this edge across all subgraphs
            edge_probs = edge_predictions[:, i]
            
            # Calculate margin (difference between mean and second most common prediction)
            mean_prob = torch.mean(edge_probs)
            # For binary case, second prediction is 1 - mean_prob
            second_prob = 1 - mean_prob
            margin = abs(mean_prob - second_prob)
            margins[i] = margin
        
        # Get final predictions
        final_predictions = torch.mean(edge_predictions, dim=0)
        binary_predictions = (final_predictions > 0.5).float()
        
        return binary_predictions, margins
    
