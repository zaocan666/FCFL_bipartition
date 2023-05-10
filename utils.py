# utils functions
import numpy as np 
import random
import os
from time import time
import pickle
import pdb
import json
import torch
import logging

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def get_random_dir_name():
    import string
    from datetime import datetime
    dirname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vocab = string.ascii_uppercase + string.ascii_lowercase + string.digits
    dirname = dirname + '-' + ''.join(random.choice(vocab) for _ in range(8))
    return dirname

    
### ready for rewright
def concave_fun(x, delta_l, delta_g):

    def f1(x):
        n = len(x)
        dx = np.linalg.norm(x - 1. / np.sqrt(n))
        return 1 - np.exp(-dx**2)

    def f2(x):
        n = len(x)
        dx = np.linalg.norm(x + 1. / np.sqrt(n))
        return 1 - np.exp(-dx**2)

    f1_dx = grad(f1)
    f2_dx = grad(f2)    

    """
    return the function values and gradient values
    """
    return np.stack([f1(x), f2(x)]), np.stack([f1_dx(x), f2_dx(x)])



def construct_log(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    os.makedirs(args.log_dir, exist_ok = True)
    handler = logging.FileHandler(os.path.join(args.log_dir ,args.log_name))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler) 
    
    return logger



def read_data(train_data_dir, test_data_dir):

    clients = []
    groups = []
    train_data = {}
    test_data = {}

    if "eicu" in train_data_dir:    
        train_files = os.listdir(train_data_dir)
        train_files = [f for f in train_files if f.endswith('.npy')]
        for f in train_files:
            file_path = os.path.join(train_data_dir,f)
            cdata = np.load(file_path, allow_pickle=True).tolist()
            train_data.update(cdata['user_data'])

        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.npy')]
        for f in test_files:
            file_path = os.path.join(test_data_dir,f)
            cdata = np.load(file_path, allow_pickle=True).tolist()
            test_data.update(cdata['user_data'])        


    elif "adult" in train_data_dir:
        train_files = os.listdir(train_data_dir)
        train_files = [f for f in train_files if f.endswith('.json')]
        for f in train_files:
            file_path = os.path.join(train_data_dir,f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            train_data.update(cdata['user_data'])

        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.json')]
        for f in test_files:
            file_path = os.path.join(test_data_dir,f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            test_data.update(cdata['user_data'])

    elif "health" in train_data_dir:
        train_files = os.listdir(train_data_dir)
        train_files = [f for f in train_files if f.endswith('.npy')]
        for f in train_files:
            file_path = os.path.join(train_data_dir, f)
            cdata = np.load(file_path, allow_pickle=True).tolist()
            train_data.update(cdata['user_data'])

        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.npy')]
        for f in test_files:
            file_path = os.path.join(test_data_dir, f)
            cdata = np.load(file_path, allow_pickle=True).tolist()
            test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


def get_multi_group_disparity(ys_pre, y, hat_ys, A, group_num, disparity_type, delta_g):
    disparities = []
    deri_disparities = []
    assert A.max()<group_num
    for i in range(group_num):
        if (A==i).sum()==0:
            continue
        for j in range(i+1, group_num):
            if (A==j).sum()==0:
                continue
            ys_pre_i = ys_pre[A==i]
            ys_pre_j = ys_pre[A==j]
            hat_ys_i = hat_ys[A==i]
            hat_ys_j = hat_ys[A==j]
            if disparity_type == "DP":
                disparity_ij = torch.sum(hat_ys_i)/hat_ys_i.shape[0] - \
                    torch.sum(hat_ys_j)/hat_ys_j.shape[0]
                disparity_ij = torch.abs(disparity_ij)

                deri_disparity_ij = torch.sum(torch.sigmoid(10 * ys_pre_i))/ys_pre_i.shape[0] - \
                    torch.sum(torch.sigmoid(10 * ys_pre_j))/ys_pre_j.shape[0]
                deri_disparity_ij = torch.abs(deri_disparity_ij)
            elif disparity_type == "Eoppo":
                raise NotImplementedError()
            
            disparities.append(disparity_ij.item())
            deri_disparities.append(deri_disparity_ij)

    deri_disparities_max = max([x.item() for x in deri_disparities])
    deri_disparity_surrogate = 0
    for ind in range(len(deri_disparities)):
        deri_disparity_surrogate += torch.exp((deri_disparities[ind]-deri_disparities_max)/delta_g)
    deri_disparity_surrogate = delta_g*torch.log(deri_disparity_surrogate)+deri_disparities_max
    return max(disparities), deri_disparity_surrogate, disparities, deri_disparities

def optim_f(model, lr):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0., weight_decay=1e-4)

def div_norm(l):
    norm_ = torch.norm(l, p=2, dim=1, keepdim=True)
    norm_[norm_<1e-4] = 1
    l = l/norm_
    return l
