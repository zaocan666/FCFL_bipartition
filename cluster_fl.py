import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
import copy

from hco_lp import HCO_LP
from po_lp import PO_LP
from utils import optim_f

def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = source1
            s2 = source2
            angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)

    return angles.numpy()

class Cluster_manager():
    def __init__(self, client_num, model:torch.nn.Module, lr, eps_cluster_global, eps_cluster_local, eps):
        self.eps_cluster_global = eps_cluster_global
        self.eps_cluster_local = eps_cluster_local
        self.eps = eps
        self.lr = lr

        self.cluster_indices = [np.arange(client_num).astype("int")]
        self.client_cluster_inds = [0]*client_num
        self.models = [model]
        if torch.cuda.is_available():
            self.models[0].cuda()
        self.optims = [optim_f(self.models[0], self.lr)]
    
        self.hco_lp_solvers = [HCO_LP(client_num=client_num, eps = eps)]
        self.po_lp_solvers = [PO_LP(client_num=client_num,  eps = eps[0])]

    def whether_split_flag(self, grad_global, grads_performance, epoch):
        grad_global_norm = torch.norm(grad_global).item()
        grad_performance_max_norm = torch.norm(grads_performance, dim=1).max().item()
        if grad_global_norm<self.eps_cluster_global and grad_performance_max_norm>self.eps_cluster_local and len(grads_performance)>=2 and epoch>-1:
            split_flag = True
        else:
            split_flag = False
        
        return split_flag, grad_global_norm, grad_performance_max_norm
    
    def set_client_cluster_inds(self):
        for cluster_ind, cluster in enumerate(self.cluster_indices):
            for i in cluster:
                self.client_cluster_inds[i] = cluster_ind

    def set_solvers(self):
        hco_lp_solvers = []
        po_lp_solvers = []
        for cluster_ind, cluster in enumerate(self.cluster_indices):
            hco_lp_solvers.append(HCO_LP(client_num=len(cluster), eps = self.eps))
            po_lp_solvers.append(PO_LP(client_num=len(cluster), eps = self.eps[0]))
        self.hco_lp_solvers = hco_lp_solvers
        self.po_lp_solvers = po_lp_solvers

    def split_clients(self, cluster_idx, client_inds, grads_performance, grads_disparity):
        grads_cat = torch.cat([grads_performance, grads_disparity], dim=1)
        similarities = pairwise_angles(grads_cat)

        clustering = AgglomerativeClustering(n_clusters=2, affinity="precomputed", linkage="complete").fit(-similarities)
        c1 = client_inds[clustering.labels_==0]
        c2 = client_inds[clustering.labels_==1]

        self.cluster_indices.append(c1)
        self.cluster_indices.append(c2)
        for i in range(2):
            self.models.append(copy.deepcopy(self.models[cluster_idx]))
            self.optims.append(optim_f(self.models[-1], self.lr))
        
        self.hco_lp_solvers.append(HCO_LP(client_num=len(c1), eps = self.eps))
        self.hco_lp_solvers.append(HCO_LP(client_num=len(c2), eps = self.eps))
        self.po_lp_solvers.append(PO_LP(client_num=len(c1), eps = self.eps[0]))
        self.po_lp_solvers.append(PO_LP(client_num=len(c2), eps = self.eps[0]))

        self.cluster_indices.pop(cluster_idx)
        self.models.pop(cluster_idx)
        self.optims.pop(cluster_idx)
        self.hco_lp_solvers.pop(cluster_idx)
        self.po_lp_solvers.pop(cluster_idx)

        for i in c1:
            self.client_cluster_inds[i] = len(self.cluster_indices)-2
        for i in c2:
            self.client_cluster_inds[i] = len(self.cluster_indices)-1

        return c1, c2
