import cvxpy as cp
import cvxopt
import torch
from utils import div_norm

class PO_LP(object): # hard-constrained optimization

    def __init__(self, client_num, eps):
        cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"
        self.client_num = client_num # the dimension of \theta
        self.eps = eps # the error bar of the optimization process eps1 < g
        self.grad_d = cp.Parameter((2*client_num, 1))       # [d_l1, d_l2, ..., d_g1, d_g1,...] * d_l .
        self.l_g = cp.Parameter(( 2*client_num, 2*client_num)) # [d_l1, d_l2, ..., d_g1, d_g1,...] * [d_l1, d_l2, ..., d_g1, d_g1,...]
        self.alpha = cp.Variable((1, 2*client_num))    # Variable to optimize
         # disparities has been satisfies, in this case we only maximize the performance
        
        obj_dom = cp.Maximize(cp.sum(self.alpha @  self.grad_d))
        constraints_dom = [self.alpha >= 0, cp.sum(self.alpha) == 1,
                            self.alpha @ self.l_g >=0]

        self.prob_dom = cp.Problem(obj_dom, constraints_dom)  # LP balance

        self.gamma = 0     # Stores the latest Optimum value of the LP problem
        self.disparity = 0     # Stores the latest maximum of selected K disparities


    def get_alpha(self, grads):
        # grads [2*client_num, d]
        grads = div_norm(grads)
        grad_l = torch.mean(grads[:self.client_num], dim = 0, keepdim=True) # [1, d]
        
        self.grad_d.value = (grads @ grad_l.t()).cpu().numpy()
        self.l_g.value = (grads @ grads.t()).cpu().numpy()
        self.gamma = self.prob_dom.solve(solver=cp.GLPK, verbose=False)

        alpha = self.alpha.value
        if torch.cuda.is_available():
            alpha = torch.from_numpy(alpha.reshape(-1)).cuda()
        else:
            alpha = torch.from_numpy(alpha.reshape(-1))
        alpha = alpha.float()
        d = -alpha.view(1, -1) @ grads
        return alpha, self.gamma, d


