import cvxpy as cp
import cvxopt
import torch
from utils import div_norm

class HCO_LP(object): # hard-constrained optimization

    def __init__(self, client_num, eps):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.client_num = client_num # the dimension of \theta
        self.eps = eps # the error bar of the optimization process [eps1 < g, eps2 < delta1, eps3 < delta2]
        self.Ca1 = cp.Parameter((self.client_num+1,1))       # [d_l, d_g1, d_g2,...] * d_l
        self.Ca2 = cp.Parameter((self.client_num+1,1))

        self.alpha = cp.Variable((1,self.client_num+1))     # Variable to optimize
         # disparities has been satisfies, in this case we only maximize the performance
        obj_dom = cp.Maximize(self.alpha @  self.Ca1) 
        obj_fair = cp.Maximize(self.alpha @ self.Ca2)


        constraints_dom = [self.alpha >= 0, cp.sum(self.alpha) == 1]
        constraints_fair = [self.alpha >= 0, cp.sum(self.alpha) == 1,
                            self.alpha @ self.Ca1 >= 0]

        self.prob_dom = cp.Problem(obj_dom, constraints_dom)  # LP balance
        self.prob_fair = cp.Problem(obj_fair, constraints_fair)

        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.disparity = 0     # Stores the latest maximum of selected K disparities


    def get_alpha(self, max_disparity, d_l, d_g):
        # d_l [1, d]
        # d_g [n, d]
        d_l = div_norm(d_l)
        d_g = div_norm(d_g)

        d_ls = torch.cat((d_l, d_g), dim=0)
        if max_disparity<= self.eps[0]: # [l, g] disparities < eps0
            self.Ca1.value = (d_ls @ d_l.t()).cpu().numpy()
            self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "dom"
        else:
            d_g_mean = torch.mean(d_g, dim=0, keepdim=True) # [1, d]
            self.Ca1.value = (d_ls @ d_l.t()).cpu().numpy() 
            self.Ca2.value = (d_ls @ d_g_mean.t()).cpu().numpy() 
            self.gamma = self.prob_fair.solve(solver=cp.GLPK, verbose=False)
            self.last_move = "fair"
        alpha = self.alpha.value
        if torch.cuda.is_available():
            alpha = torch.from_numpy(alpha.reshape(-1)).cuda()
        else:
            alpha = torch.from_numpy(alpha.reshape(-1))
        alpha = alpha.float()
        d = -alpha.view(1, -1) @ d_ls

        return alpha, d
