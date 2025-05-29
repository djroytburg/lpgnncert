import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch import nn
from torch_geometric.nn import APPNP, GAE
from torch_geometric.data import Data

from gnncert.hash_agent import HashAgent, RobustEdgeClassifier
from lp_models.BaseLP import BaseLp


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index,scaling_factor=1.8):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)
        self.scaling_factor=scaling_factor

    def forward(self, x, edge_index, not_prop=0):
        x_ = self.linear1(x)
        x_ = self.propagate(x_, edge_index)

        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1) * self.scaling_factor
        x = self.propagate(x, edge_index)
        return x, x_
class VGNAE(GAE):
    def __init__(self, encoder, decoder = None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=10)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def forward(self, data, edge_index):
        '''
        Predict edge probabilities on edge_index.
        Assumption: every node in edge_index is in data.x.
        '''
        return self.decoder(self.encode(data.x, data.train_pos_edge_index), edge_index)
    
    def kl_loss(self, mu=None,
                logstd= None) :
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
    def test(self, data, split='val'):
        from sklearn.metrics import average_precision_score, roc_auc_score
        if split=='val':       
            pos_edge_index = data.val_pos_edge_index
            neg_edge_index = data.val_neg_edge_index
        elif split=='test':
            pos_edge_index = data.test_pos_edge_index
            neg_edge_index = data.test_neg_edge_index
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'val' or 'test'.")
        pos_y = data.x.new_ones(pos_edge_index.size(1))
        neg_y = data.x.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self(data, pos_edge_index)
        neg_pred = self(data, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        acc= ((pred > 0.5) == y).mean()
        recall = ((pred > 0.5) * y).sum() / y.sum()
        precision = ((pred > 0.5) * y).sum() / (pred > 0.5).sum()
        f1=2*recall*precision/(recall+precision)
        auc=roc_auc_score(y, pred)

        return {'auc': auc, 'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1},[pos_pred, neg_pred]


class VGNAE_LP(BaseLp):
    #初始化
    def __init__(self, data, embedding_dim, device, p=0.4, T=12):
        super().__init__()
        self.robust = False
        try:
            self.model = VGNAE(Encoder(data.x.size()[1], embedding_dim, data.train_pos_edge_index)).to(device)
        except AttributeError:
            import pickle
            self.robust = True
            pickle.dump(data, open('hybridized_data.pkl','wb'))
            size = data.graphs[0].x.size()[1]
            self.model = VGNAE(Encoder(size, embedding_dim, data.graphs[0].train_pos_edge_index)).to(device)
            self.T = T; self.p = p
            self.hash_agent = HashAgent(T=self.T,p=self.p)
        self.device = device
    
    def test(self, data, split='val', subgraphs=None):
        from sklearn.metrics import average_precision_score, roc_auc_score
        if split=='val':       
            pos_edge_index = data.val_pos_edge_index
            neg_edge_index = data.val_neg_edge_index
        elif split=='test':
            pos_edge_index = data.test_pos_edge_index
            neg_edge_index = data.test_neg_edge_index
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'val' or 'test'.")
        pos_y = data.x.new_ones(pos_edge_index.size(1))
        neg_y = data.x.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        if self.robust:
            try:
                # print(split)
                # print("pos")
                pos_pred = self.classifier(data, pos_edge_index, subgraphs=subgraphs, split=split)
                # print("neg")
                neg_pred = self.classifier(data, neg_edge_index, subgraphs=subgraphs, split=split)
            except AttributeError:
                print(type(pos_edge_index), type(neg_edge_index), type(subgraphs[0]))
                print(pos_edge_index, neg_edge_index, subgraphs[0])
                raise ValueError('subgraphs is not None')
        else:
            pos_pred = self.model(data, pos_edge_index)
            neg_pred = self.model(data, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        acc= ((pred > 0.5) == y).mean()
        recall = ((pred > 0.5) * y).sum() / y.sum()
        precision = ((pred > 0.5) * y).sum() / (pred > 0.5).sum()
        f1=2*recall*precision/(recall+precision)
        auc=roc_auc_score(y, pred)

        return {'auc': auc, 'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1},[pos_pred, neg_pred]

    def train(self,data,optimizer,epochs):
        best_val_result = 0
        best_val_result_auc = 0
        best_test_result = 0
        best_model = copy.deepcopy(self.model)
        best_scores = None
        if self.robust:
            graphs = data
            test_graph = graphs.graphs[0]
        else:
            graphs = Data(graphs=[data], device=self.device)
            test_graph = data
        for epoch in tqdm.tqdm(range(epochs)):
            self.model.train()
            for i, data in enumerate(graphs.graphs):
                # if i > 0:
                #     continue
                optimizer.zero_grad()
                z = self.model.encode(data.x.to(self.device), data.train_pos_edge_index.to(self.device))
                loss = self.model.recon_loss(z, data.train_pos_edge_index.to(self.device))
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    if self.robust:
                        self.classifier = RobustEdgeClassifier(self.model, self.hash_agent)
                        val_result,val_score= self.test(test_graph.to(self.device), split='val', subgraphs=graphs.graphs)
                        test_result,test_score = self.test(test_graph.to(self.device), split='test', subgraphs=graphs.graphs)
                    else:
                        val_result,val_score= self.model.test(test_graph.to(self.device), split='val')
                        test_result,test_score = self.model.test(test_graph.to(self.device), split='test')
                    scores=[val_score,test_score]
                    if val_result['auc'] > best_val_result_auc:
                        best_val_result = val_result
                        best_val_result_auc = val_result['auc']
                        best_test_result = test_result
                        best_model = copy.deepcopy(self.model)
                        best_scores = scores
        self.model=best_model
        return best_val_result,best_test_result ,best_scores
    def get_embedding(self,data,optimizer=None,epochs=None):
        best_val_result,best_test_result ,best_scores=self.train(data,optimizer,epochs)
        self.model.eval()
        embeddings=self.model.encode(self.data.x, self.data.train_pos_edge_index).detach()
        return F.normalize(embeddings)