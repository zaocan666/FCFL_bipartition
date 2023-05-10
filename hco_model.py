import torch
import torch.nn as nn
from torch.autograd import Variable
from dataload import LoadDataset
import os
import numpy as np
import json
import torch.nn.functional as F 
import pickle
from sklearn.metrics import roc_auc_score
import copy

from utils import get_multi_group_disparity, optim_f
from cluster_fl import Cluster_manager
from utils import optim_f

class RegressionTrain(torch.nn.Module):

    def __init__(self, model, disparity_type = "DP", dataset  = "adult", sensitive_group=4, baseline_type='none'):
        super(RegressionTrain, self).__init__()
        self.model = model
        self.loss = nn.BCELoss()
        self.disparity_type = disparity_type
        self.dataset = dataset
        self.sensitive_group = sensitive_group
        self.baseline_type = baseline_type

    def forward(self, x, y, A, delta_g, mu=None):
        ys_pre = self.model(x).flatten()
        ys = torch.sigmoid(ys_pre)
        hat_ys = (ys >=0.5).float()
        
        task_loss_as = []
        if self.baseline_type=='fedminmax':
            task_loss = 0
            for a in range(self.sensitive_group):
                loss_a = self.loss(ys[A==a], y[A==a])
                if not torch.isnan(loss_a):
                    task_loss += loss_a*mu[a]
                task_loss_as.append(loss_a.item())

            task_loss_as = np.array(task_loss_as)
        else:
            task_loss = self.loss(ys, y)
        
        accs = torch.mean((hat_ys == y).float()).item()
        aucs = roc_auc_score(y.cpu(), ys.clone().detach().cpu())

        disparity_max, deri_disparity_surrogate, disparities, deri_disparities = \
                get_multi_group_disparity(ys_pre, y, hat_ys, A, self.sensitive_group, self.disparity_type, delta_g)
        
        pred_dis = deri_disparity_surrogate # derivable disparity
        disparitys = disparity_max # disparity
        return task_loss, accs, aucs, pred_dis, disparitys, ys, task_loss_as
            
    def randomize(self):
        self.model.apply(weights_init)



def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.weight.data *= 0.1

class RegressionModel(torch.nn.Module):
    def __init__(self, n_feats, n_hidden):
        super(RegressionModel, self).__init__()

        self.logstic = nn.Linear(n_feats, n_hidden)
        self.logstic2 = nn.Linear(n_hidden, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        out1 = self.logstic(x)
        out1 = self.act(out1)
        out1 = self.logstic2(out1)
        return out1


class MODEL(object):
    def __init__(self, args, logger, writer):
        super(MODEL, self).__init__()


        self.dataset = args.dataset
        self.different_eps = bool(args.different_eps)
        if not self.different_eps:
            self.eps = args.eps
        else:
            args.eps[0] = 0.0
            self.eps = [0.0]

        self.max_epoch1 = args.max_epoch_stage1
        self.max_epoch2 = args.max_epoch_stage2
        self.ckpt_dir = args.ckpt_dir
        self.global_epoch = args.global_epoch
        self.log_pickle_dir = args.log_dir
        self.per_epoches = args.per_epoches
        self.deltas = np.array([0.,0.])
        self.deltas[0] = args.delta_l
        self.deltas[1] = args.delta_g
        self.eval_epoch = args.eval_epoch
        self.logger = logger
        self.logger.info(str(args))
        self.n_linscalar_adjusts = 0
        self.done_dir = args.done_dir
        self.writer = writer
        self.uniform = args.uniform
        self.performence_only = args.uniform
        self.policy = args.policy
        self.disparity_type = args.disparity_type
        self.baseline_type = args.baseline_type
        self.weight_fair = args.weight_fair
        self.sensitive_attr = args.sensitive_attr
        self.sensitive_group = args.sensitive_group
        self.weight_eps = args.weight_eps
        self.fedminmax_mu_lr = args.fedminmax_mu_lr

        self.data_load(args)

        model_tmp = RegressionTrain(RegressionModel(self.n_feats, args.n_hiddens), args.disparity_type, args.dataset, args.sensitive_group, self.baseline_type)
        if self.baseline_type == "none":
            self.cluster_manager = Cluster_manager(client_num=self.n_clients, 
                            model=model_tmp,
                            lr=args.step_size,
                            eps_cluster_global=args.eps_cluster_global,
                            eps_cluster_local=args.eps_cluster_local,
                            eps=self.eps)
        elif self.baseline_type == "fedave_fair" or self.baseline_type == "fedminmax":
            self.model = model_tmp
            if torch.cuda.is_available():
                self.model.cuda()
            self.optim = optim_f(self.model, lr=args.step_size)

        self.log_train = dict()
        self.log_test = dict()

        _, n_params = self.getNumParams(model_tmp.parameters())
        self.logger.info('param num: %d'%n_params)

        if int(args.load_epoch) != 0:
            self.model_load(str(args.load_epoch), args)

        self.commandline_save(args)
        self.last_model_pth = None

        self.init_mu = self.get_init_mu()
        self.mu = None
        if self.baseline_type=='fedminmax':
            self.mu = self.init_mu

    def commandline_save(self, args):
        with open(args.commandline_file, "w") as f:
            json.dump(args.__dict__, f, indent =2)

    def getNumParams(self, params):
        numParams, numTrainable = 0, 0
        for param in params:
            npParamCount = np.prod(param.data.shape)
            numParams += npParamCount
            if param.requires_grad:
                numTrainable += npParamCount
        return numParams, numTrainable


    def model_load(self, ckptname, args):

        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                self.logger.info("=> no checkpoint found")
                exit()
            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])
        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            cluster_indices = checkpoint['cluster_indices']
            self.cluster_manager.cluster_indices = cluster_indices
            self.cluster_manager.set_client_cluster_inds()
            self.cluster_manager.set_solvers()

            for i in range(len(cluster_indices)):
                if i<len(self.cluster_manager.models):
                    model = self.cluster_manager.models[i]
                else:
                    model = copy.deepcopy(self.cluster_manager.models[0])
                model.load_state_dict(checkpoint['model_%d'%i])
                optim = optim_f(model, args.step_size)

                if i<len(self.cluster_manager.models):
                    self.cluster_manager.models[i] = model
                    self.cluster_manager.optims[i] = optim
                else:
                    self.cluster_manager.models.append(model)
                    self.cluster_manager.optims.append(optim)

            self.logger.info("=> loaded checkpoint '{} (epoch {})'".format(filepath, self.global_epoch))

        else:
            self.logger.info("=> no checkpoint found at '{}'".format(filepath))


    def model_save(self, ckptname = None):
        if self.baseline_type == "none":
            states = {'epoch':self.global_epoch,
                    'cluster_indices':self.cluster_manager.cluster_indices}
            for i in range(len(self.cluster_manager.models)):
                states['model_%d'%i] = self.cluster_manager.models[i].state_dict()
                states['optim_%d'%i] = self.cluster_manager.optims[i].state_dict()

        elif self.baseline_type == "fedave_fair" or self.baseline_type == "fedminmax":
            states = {'epoch':self.global_epoch,
                  'model':self.model.state_dict(),
                  'optim':self.optim.state_dict()}

        if ckptname == None:
            ckptname = str(self.global_epoch)

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        os.makedirs(self.ckpt_dir, exist_ok = True)
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        self.logger.info("=> saved checkpoint '{}' (epoch {})".format(filepath, self.global_epoch))

        if self.last_model_pth:
            os.system('rm ' + self.last_model_pth)
        self.last_model_pth = filepath


    def data_load(self, args):
        self.client_train_loaders, self.client_test_loaders, n_feats = LoadDataset(args)
        self.n_feats = n_feats
        self.n_clients = len(self.client_train_loaders)
        self.iter_train_clients = [enumerate(i) for i in self.client_train_loaders]
        self.iter_test_clients = [enumerate(i) for i in self.client_test_loaders]
        # 0 phd, 1 non-phd

        train_dataset_len = [len(self.client_train_loaders[i].dataset) for i in range(len(self.client_train_loaders))]
        test_dataset_len = [len(self.client_test_loaders[i].dataset) for i in range(len(self.client_train_loaders))]
        self.logger.info('dataset train len: '+str(train_dataset_len))
        self.logger.info('dataset test len: '+str(test_dataset_len))

    def valid_stage1(self,  if_train = False, epoch = -1):
        mu = self.init_mu
        with torch.no_grad():
            losses = []
            accs = []
            diss = []
            pred_diss = []
            aucs = []
            if if_train:
                loader = self.client_train_loaders
            else:
                loader = self.client_test_loaders
            for client_idx in range(self.n_clients):
                client_test_loader = loader[client_idx]
                if self.baseline_type == "none":
                    cluster_idx = self.cluster_manager.client_cluster_inds[client_idx]
                    model = self.cluster_manager.models[cluster_idx]
                elif self.baseline_type == "fedave_fair" or self.baseline_type == "fedminmax":
                    model = self.model
                model.eval()

                valid_loss = []
                valid_accs = []
                valid_diss = []
                valid_pred_dis = []
                valid_auc = []
                for it, (X, Y, A) in enumerate(client_test_loader):
                    X = X.float()
                    Y = Y.float()
                    A = A.float()
                    if torch.cuda.is_available():
                        X = X.cuda()
                        Y = Y.cuda()
                        A = A.cuda()
                    loss, acc, auc, pred_dis, disparity, pred_y, _ = model(X, Y, A, self.deltas[1], mu)
                    valid_loss.append(loss.item())
                    valid_accs.append(acc)  
                    valid_diss.append(disparity)
                    valid_pred_dis.append(pred_dis.item())
                    valid_auc.append(auc)
                assert len(valid_auc)==1
                losses.append(np.mean(valid_loss))
                accs.append(np.mean(valid_accs))
                diss.append(np.mean(valid_diss))
                pred_diss.append(np.mean(valid_pred_dis))
                aucs.append(np.mean(valid_auc))
            self.logger.info("Valid is_train: {}, epoch: {}, ACC avg: {}, disparity avg: {}, loss: {}, accuracy: {}, auc: {}, disparity: {}, pred_disparity: {}".format(
                if_train, self.global_epoch, np.mean(accs), np.mean(diss), losses, accs, aucs, diss, pred_diss))
            self.log_test[str(epoch)] = { "client_losses": losses, "pred_client_disparities": pred_diss, "client_accs": accs, "client_aucs": aucs, "client_disparities": diss, "max_losses": [max(losses), max(diss)]}

            if self.baseline_type == "none":
                self.log_test[str(epoch)]["client_cluster_inds"] = self.cluster_manager.client_cluster_inds
                
            if if_train:
                for i, item in enumerate(losses):
                    self.writer.add_scalar("valid_train/loss_:"+str(i),  item , epoch)
                    self.writer.add_scalar("valid_trains/acc_:"+str(i),  accs[i], epoch)
                    self.writer.add_scalar("valid_trains/auc_:"+str(i),  aucs[i], epoch)
                    self.writer.add_scalar("valid_trains/disparity_:"+str(i),  diss[i], epoch)
                    self.writer.add_scalar("valid_trains/pred_disparity_:"+str(i),  pred_diss[i], epoch)
                    self.writer.add_scalar('valid_trains/acc_mean', np.mean(accs), epoch)
                    self.writer.add_scalar('valid_trains/disparity_mean', np.mean(diss), epoch)
            else:
                for i, item in enumerate(losses):
                    self.writer.add_scalar("valid_test/loss_:"+str(i),  item , epoch)
                    self.writer.add_scalar("valid_test/acc_:"+str(i),  accs[i], epoch)   
                    self.writer.add_scalar("valid_test/auc_:"+str(i),  aucs[i], epoch) 
                    self.writer.add_scalar("valid_test/disparity_:"+str(i),  diss[i], epoch)   
                    self.writer.add_scalar("valid_test/pred_disparity_:"+str(i),  pred_diss[i], epoch)
                    self.writer.add_scalar('valid_test/acc_mean', np.mean(accs), epoch)
                    self.writer.add_scalar('valid_test/disparity_mean', np.mean(diss), epoch)
            return losses, accs, diss, pred_diss, aucs
   

    def soften_losses(self, losses, delta):


        losses_list = torch.stack(losses)
        loss = torch.max(losses_list)
        alphas = F.softmax((losses_list - loss)/delta, dim=0)
        alpha_without_grad = (Variable(alphas.data.clone(), requires_grad=False)) 
        return alpha_without_grad, loss


    def train(self):
        
        if self.baseline_type == "none":
            if self.policy == "alternating":
                start_epoch = self.global_epoch
                for epoch in range(start_epoch , self.max_epoch1 + self.max_epoch2):
                    if int(epoch/self.per_epoches) %2 == 0:
                        self.train_stage1(epoch)
                    else:
                        self.train_stage2(epoch)

            elif self.policy == "two_stage":
                if self.uniform:
                    self.performence_only  = True
                else:
                    self.performence_only  = False
                start_epoch = self.global_epoch
                for epoch in range(start_epoch, self.max_epoch1):
                    self.train_stage1(epoch)

                for epoch in range(self.max_epoch1, self.max_epoch2 + self.max_epoch1):
                    self.train_stage2(epoch)


        elif self.baseline_type == "fedave_fair" or self.baseline_type == "fedminmax":
            start_epoch = self.global_epoch
            for epoch in range(start_epoch, self.max_epoch2 + self.max_epoch1):
                self.train_fed(epoch)


    def save_log(self):
        with open(os.path.join(self.log_pickle_dir, "train_log.pkl"), "wb") as f:
            pickle.dump(self.log_train, f)
        with open(os.path.join(self.log_pickle_dir, "test_log.pkl"), "wb") as f:
            pickle.dump(self.log_test, f)    
        os.makedirs(self.done_dir, exist_ok = True) 
        self.logger.info("logs have been saved")   

    def get_init_mu(self):
        mu = np.zeros(self.sensitive_group)
        for client_idx in range(self.n_clients):
            try:
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            except StopIteration:
                self.iter_train_clients[client_idx] = enumerate(
                    self.client_train_loaders[client_idx])
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            
            for a in range(mu.shape[0]):
                mu[a] += (A==a).sum()

        mu = mu/mu.sum()
        return mu


    def train_fed(self, epoch):

        self.model.train()
        self.optim.zero_grad()
        losses_data = []
        disparities_data = []
        pred_disparities_data = []
        accs_data = []
        aucs_data = []
        client_losses = []
        client_disparities = []

        loss_as_all = []
        for client_idx in range(self.n_clients):
            try:
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            except StopIteration:
                self.iter_train_clients[client_idx] = enumerate(
                    self.client_train_loaders[client_idx])
                _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
            X = X.float()
            Y = Y.float()
            A = A.float()
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
                A = A.cuda()

            loss, acc, auc, pred_dis, dis, pred_y, loss_as = self.model(X, Y, A, self.deltas[1], self.mu)
            loss_as_all.append(loss_as)

############################################################## GPU version

            client_losses.append(loss)
            client_disparities.append(pred_dis)
            losses_data.append(loss.item())
            disparities_data.append(dis)
            pred_disparities_data.append(pred_dis.item())
            accs_data.append(acc)
            aucs_data.append(auc)

        loss_max_performance = max(losses_data)
        loss_max_disparity = disparities_data[np.argmax(disparities_data)]
        self.logger.info("{}, epoch: {}, ACC avg: {}, disparity avg: {}, all client loss: {}, all pred client disparities: {}, all client disparities: {}, all client accs: {}, all client aucs: {},  all max loss: {}, mu: {}".format(
                    self.baseline_type, self.global_epoch, np.mean(accs_data), np.mean(disparities_data), 
                    losses_data, pred_disparities_data, disparities_data, accs_data, aucs_data,  [loss_max_performance, loss_max_disparity], self.mu))
        
        self.log_train[str(epoch)] = {"stage": 1, "client_losses": losses_data, "pred_client_disparities": pred_disparities_data, "client_disparities": disparities_data,
                                      "client_accs": accs_data, "client_aucs": aucs_data, "max_losses": [loss_max_performance, loss_max_disparity]}

        for i, loss in enumerate(losses_data):
            self.writer.add_scalar("train/1_loss_" + str(i), loss, epoch)
            self.writer.add_scalar(
                "train/disparity_" + str(i), disparities_data[i], epoch)
            self.writer.add_scalar(
                "train/pred_disparity_" + str(i), pred_disparities_data[i], epoch)
            self.writer.add_scalar(
                "train/acc_" + str(i), accs_data[i], epoch)
            self.writer.add_scalar(
                "train/auc_" + str(i), aucs_data[i], epoch)

        self.writer.add_scalar('train/acc_mean', np.mean(accs_data), epoch)
        self.writer.add_scalar('train/disparity_mean', np.mean(disparities_data), epoch)

        self.optim.zero_grad()
        weighted_loss1 = torch.sum(torch.stack(client_losses))
        if self.baseline_type=='fedave_fair':
            weighted_loss2 = torch.sum(torch.stack(client_disparities)) * self.weight_fair
            weighted_loss = weighted_loss1 + weighted_loss2
        elif self.baseline_type=='fedminmax':
            weighted_loss = weighted_loss1
        weighted_loss.backward()
        self.optim.step()

        if self.baseline_type=='fedminmax':
            loss_as_all = np.nanmean(loss_as_all, axis=0)
            self.mu += loss_as_all*self.fedminmax_mu_lr
            self.mu = self.euclidean_proj_simplex(torch.Tensor(self.mu)).numpy()

        # Calculate and record performance
        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            losses, accs, client_disparities, pred_dis, aucs = self.valid_stage1(
                if_train=False, epoch=epoch)
        self.global_epoch += 1

    def euclidean_proj_simplex(self, v, s=1):
        eps = 1e-3
        assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
        n, = v.shape  # will raise ValueError if v is not 1-D
        # check if we are already on the simplex
        if v.sum() == s and (v >= eps).all():
            # best projection: itself!
            return v
        # get the array of cumulative sums of a sorted (decreasing) copy of v
        u = torch.flip(torch.sort(v)[0],dims=(0,))
        cssv = torch.cumsum(u,dim=0)
        # get the number of > 0 components of the optimal solution
        non_zero_vector = torch.nonzero(u * torch.arange(1, n+1) > (cssv - s), as_tuple=False)

        if len(non_zero_vector) == 0:
            rho=0
        else:
            rho = non_zero_vector[-1].squeeze()
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - s) / (rho + 1.0)
        # compute the projection by thresholding v using theta
        w = (v - theta).clamp(min=0)

        w_zeros = (w <= 1e-3)
        if w_zeros.sum() > 0:
            w[w_zeros] = 1e-3
            w /= w.sum()

        return w  

    def train_stage1(self, epoch):

        losses_data = np.zeros([self.n_clients])
        disparities_data = np.zeros([self.n_clients])
        pred_disparities_data = np.zeros([self.n_clients])
        accs_data = np.zeros([self.n_clients])
        aucs_data = np.zeros([self.n_clients])

        loss_max_performance_all = []
        loss_max_disparity_all = []
        grad_global_len_all = {}
        grad_performance_max_len_all = {}

        for cluster_idx, client_inds in enumerate(self.cluster_manager.cluster_indices):
            model = self.cluster_manager.models[cluster_idx]
            optim = self.cluster_manager.optims[cluster_idx]
            hco_lp = self.cluster_manager.hco_lp_solvers[cluster_idx]
            
            model.train()
            optim.zero_grad()
            grads_performance = []
            grads_disparity = []
            
            client_losses = []
            client_disparities = []

            for client_idx in client_inds:
                try:
                    _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
                except StopIteration:
                    self.iter_train_clients[client_idx] = enumerate(self.client_train_loaders[client_idx])
                    _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
                X = X.float()
                Y = Y.float()
                A = A.float()
                if torch.cuda.is_available():
                    X = X.cuda()
                    Y = Y.cuda()
                    A = A.cuda()

                loss, acc, auc, pred_dis, dis, pred_y, _ = model(X, Y, A, self.deltas[1])

                loss.backward(retain_graph=True)
                grad = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
                grad = torch.stack(grad)
                grads_performance.append(grad)
                optim.zero_grad()


                pred_dis.backward(retain_graph=True)
                if self.performence_only:
                    optim.zero_grad()
                grad = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
                grad = torch.stack(grad)
                grads_disparity.append(grad)
                optim.zero_grad()   

                if not self.different_eps:
                    client_disparities.append(pred_dis)
                    specific_eps = 0

                else:
                    if "eicu_los" in self.dataset and self.disparity_type == "DP" and self.sensitive_attr == "race":
                        base_dis = np.array([0.0149141 , 0.10728183, 0.04039593, 0.03920709, 0.04729731,
                                            0.03609734, 0.075837  , 0.02757892, 0.02468352, 0.01953895,
                                            0.05942936, 0.065047  ])
                        specific_eps = self.weight_eps * base_dis
                    else:
                        raise NotImplementedError()


                    client_disparities.append(pred_dis - specific_eps[client_idx])


                client_losses.append(loss)
                losses_data[client_idx] = loss.item()
                disparities_data[client_idx] = dis
                pred_disparities_data[client_idx] = pred_dis.item()
                accs_data[client_idx] = acc
                aucs_data[client_idx] = auc

            alphas_l, loss_max_performance = self.soften_losses(client_losses, self.deltas[0]) # alpha_l = (d surrogate_l)/(d li)
            loss_max_performance = loss_max_performance.item()
            loss_max_disparity = torch.Tensor(client_disparities).max().item()

            loss_max_performance_all.append(loss_max_performance)
            loss_max_disparity_all.append(loss_max_disparity)

            grads_performance = torch.stack(grads_performance)
            grads_disparity = torch.stack(grads_disparity)

            # Calculate the alphas from the LP solver
            alphas_l = alphas_l.view(1, -1)
            grad_l = alphas_l @ grads_performance
            grad_g = grads_disparity # [n, d]
            alpha, d = hco_lp.get_alpha(loss_max_disparity, grad_l, grad_g)

            optim.zero_grad()
            weighted_loss1 = torch.sum(torch.stack(client_losses)*alphas_l)
            loss2 = torch.stack(client_disparities)
            weighted_loss = torch.sum(torch.cat([weighted_loss1.view(1), loss2], dim=0) * alpha)
            weighted_loss.backward()
            
            grad_global = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_global.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
            grad_global = torch.stack(grad_global)

            optim.step()

            split_flag, grad_global_norm, grad_performance_max_norm = self.cluster_manager.whether_split_flag(grad_global, grads_performance, epoch)
            self.logger.info('cluster: {}, loss_max_performance: {}, loss_max_disparity: {}, alpha: {}, grad_global_len: {}, grad_performance_max_len: {}'.format(
                client_inds, loss_max_performance, loss_max_disparity, alpha.cpu().numpy(), grad_global_norm, grad_performance_max_norm))

            grad_global_len_all[str(cluster_idx)] = grad_global_norm
            grad_performance_max_len_all[str(client_idx)] = grad_performance_max_norm

            if split_flag:
                c1, c2 = self.cluster_manager.split_clients(cluster_idx, client_inds, grads_performance, grads_disparity)
                self.logger.info('cluster %s split into %s and %s'%(str(client_inds), str(c1), str(c2)))

        self.logger.info("1, epoch: {}, ACC avg: {}, disparity avg: {}, all client loss: {}, all pred client disparities: {}, all client disparities: {}, all client accs: {}, all client aucs: {}, specific eps: {}, all Deltas: {}".format(
            self.global_epoch, np.mean(accs_data), np.mean(disparities_data), losses_data, pred_disparities_data, disparities_data, accs_data, aucs_data, specific_eps, self.deltas))
        self.log_train[str(epoch)] = { "stage": 1, "client_losses": losses_data, "pred_client_disparities": pred_disparities_data,
                                        "client_disparities": disparities_data, "client_accs": accs_data, "client_aucs": aucs_data,
                                        "loss_max_performance": loss_max_performance_all, "loss_max_disparity": loss_max_disparity_all,
                                        "deltas": self.deltas, 'grad_global_len_all':grad_global_len_all, 'grad_performance_max_len_all':grad_performance_max_len_all}
        
        if self.baseline_type == "none":
            self.log_train[str(epoch)]["client_cluster_inds"] = self.cluster_manager.client_cluster_inds

        for i, loss in enumerate(losses_data):
            self.writer.add_scalar("train/1_loss_" + str(i), loss, epoch)
            self.writer.add_scalar("train/1_disparity_" + str(i), disparities_data[i], epoch)
            self.writer.add_scalar("train/1_pred_disparity_" + str(i), pred_disparities_data[i], epoch)
            self.writer.add_scalar("train/1_acc_" + str(i), accs_data[i], epoch)
            self.writer.add_scalar("train/1_auc_" + str(i), aucs_data[i], epoch)

        self.writer.add_scalar('train/1_acc_mean', np.mean(accs_data), epoch)
        self.writer.add_scalar('train/1_disparity_mean', np.mean(disparities_data), epoch)

        for i, delta in enumerate(self.deltas):
            self.writer.add_scalar("train/1_delta_" + str(i), delta, epoch)
        
        self.writer.add_scalars('train/1_grad_global_len', grad_global_len_all, epoch)
        self.writer.add_scalars('train/1_grad_performance_max_len', grad_performance_max_len_all, epoch)

        # Calculate and record performance
        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            losses, accs, client_disparities, pred_dis, aucs = self.valid_stage1(if_train = False, epoch = epoch)

        self.global_epoch+=1

    def train_stage2(self, epoch):
        
        losses_data = np.zeros([self.n_clients])
        disparities_data = np.zeros([self.n_clients])
        pred_disparities_data = np.zeros([self.n_clients])
        accs_data = np.zeros([self.n_clients])
        aucs_data = np.zeros([self.n_clients])

        grad_global_len_all = {}
        grad_performance_max_len_all = {}

        for cluster_idx, client_inds in enumerate(self.cluster_manager.cluster_indices):
            model = self.cluster_manager.models[cluster_idx]
            optim = self.cluster_manager.optims[cluster_idx]
            po_lp = self.cluster_manager.po_lp_solvers[cluster_idx]

            model.train()
            grads_performance = []
            grads_disparity = []
            client_losses = []
            client_disparities = []

            for client_idx in client_inds:

                try:
                    _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
                except StopIteration:
                    self.iter_train_clients[client_idx] = enumerate(self.client_train_loaders[client_idx])
                    _, (X, Y, A) = self.iter_train_clients[client_idx].__next__()
                X = X.float()
                Y = Y.float()
                A = A.float()
                if torch.cuda.is_available():
                    X = X.cuda()
                    Y = Y.cuda()
                    A = A.cuda()

                loss, acc, auc, pred_dis, dis, pred_y, _ = model(X, Y, A, self.deltas[1])

                loss.backward(retain_graph=True)
                grad = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
                grad = torch.stack(grad)
                grads_performance.append(grad)
                optim.zero_grad()


                pred_dis.backward(retain_graph=True)
                if self.performence_only:
                    optim.zero_grad() 
                grad = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
                grad = torch.stack(grad)
                grads_disparity.append(grad)
                optim.zero_grad()

                client_losses.append(loss)
                client_disparities.append(pred_dis)
                losses_data[client_idx] = loss.item()
                disparities_data[client_idx] = dis
                pred_disparities_data[client_idx] = pred_dis.item()
                accs_data[client_idx] = acc
                aucs_data[client_idx] = auc

            grads_disparity = torch.stack(grads_disparity)
            grads_performance = torch.stack(grads_performance)

            max_disparity = torch.stack(client_disparities).max().item()
            if max_disparity < self.eps[0]:
                grads_disparity = torch.zeros_like(grads_disparity, requires_grad= False)
            
            grads = torch.cat((grads_performance, grads_disparity), dim = 0)
            alpha, gamma, d = po_lp.get_alpha(grads)

            weighted_loss = torch.sum(torch.stack(client_losses+client_disparities) * alpha)
            weighted_loss.backward()

            grad_global = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_global.extend(Variable(param.grad.data.clone().flatten(), requires_grad=False)) 
            grad_global = torch.stack(grad_global)

            optim.step()

            split_flag, grad_global_norm, grad_performance_max_norm = self.cluster_manager.whether_split_flag(grad_global, grads_performance, epoch)
            self.logger.info('cluster: {}, max_disparity: {}, alpha: {}, grad_global_len: {}, grad_performance_max_len: {}'.format(
                client_inds, max_disparity, alpha.cpu().numpy(), grad_global_norm, grad_performance_max_norm))

            optim.zero_grad()
        
            if split_flag:
                c1, c2 = self.cluster_manager.split_clients(cluster_idx, client_inds, grads_performance, grads_disparity)
                self.logger.info('cluster %s split into %s and %s'%(str(client_inds), str(c1), str(c2)))
            
            grad_global_len_all[str(cluster_idx)] = grad_global_norm
            grad_performance_max_len_all[str(client_idx)] = grad_performance_max_norm

        self.logger.info("2, epoch: {}, ACC avg: {}, disparity avg: {}, all client loss: {}, all pred client disparities: {}, all client disparities: {}, all client accs: {}, all client aucs: {}, deltas: {}".format(
            epoch, np.mean(accs_data), np.mean(disparities_data), losses_data, pred_disparities_data, disparities_data, accs_data, aucs_data, self.deltas))

        self.log_train[str(epoch)] = { "stage": 2, "client_losses": losses_data, "pred_client_disparities": pred_disparities_data,
                                       "client_disparities": disparities_data, "client_accs": accs_data, "client_aucs": aucs_data,
                                         "max_losses": [max(losses_data), max(disparities_data)], "deltas": self.deltas,
                                         'grad_global_len_all':grad_global_len_all, 'grad_performance_max_len_all':grad_performance_max_len_all}

        if self.baseline_type == "none":
            self.log_train[str(epoch)]["client_cluster_inds"] = self.cluster_manager.client_cluster_inds

        for i, loss in enumerate(losses_data):
            self.writer.add_scalar("train/2_loss_" + str(i), loss, epoch)
            self.writer.add_scalar("train/2_disparity_" + str(i), disparities_data[i], epoch)
            self.writer.add_scalar("train/2_pred_disparity_" + str(i), pred_disparities_data[i], epoch)
            self.writer.add_scalar("train/2_acc_" + str(i), accs_data[i], epoch)
            self.writer.add_scalar("train/2_auc_" + str(i), aucs_data[i], epoch)

        self.writer.add_scalar('train/2_acc_mean', np.mean(accs_data), epoch)
        self.writer.add_scalar('train/2_disparity_mean', np.mean(disparities_data), epoch)

        self.writer.add_scalars('train/2_grad_global_len', grad_global_len_all, epoch)
        self.writer.add_scalars('train/2_grad_performance_max_len', grad_performance_max_len_all, epoch)

        if epoch == 0 or (epoch + 1) % self.eval_epoch == 0:
            losses, accs, client_disparities, pred_dis, aucs = self.valid_stage1(if_train = False, epoch = epoch)
            
        self.global_epoch+=1
