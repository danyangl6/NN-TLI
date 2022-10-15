import torch
import torch.nn as nn
import numpy as np

class STEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g):
        # g -> gs
        g_clip = torch.clamp(g, min=0, max = 1)
        gs = g_clip.clone()
        gs[gs>=0.5] = 1
        gs[gs<0.5] = 0
        return gs
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clone(grad_output)
        return grad_input


class Clip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g):
        gs = g.clone()
        gs[gs>0] = 1
        gs[gs<=0] = -1
        return gs
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.clone(grad_output)
        return grad_input


class RMinTimeWeight(object):
    def __init__(self, tau, t1, t2):
        self.t1 = t1
        self.t2 = t2
        self.tau = tau
        self.relu = nn.ReLU()
    def get_weight(self,w):
        f1 = (self.relu(w-self.t1+self.tau)-self.relu(w-self.t1))/self.tau
        f2 = (self.relu(-w+self.t2+self.tau)-self.relu(-w+self.t2))/self.tau
        w = torch.min(f1,f2)
        return w


class SparseMax(object):
    def __init__(self, beta, a, dim):
        self.beta = beta
        self.a = a
        self.dim = dim
    def f(self, r):
        robust = torch.exp(self.beta * r)-self.a
        return robust
    def forward(self, r):
        r = self.f(r)
        r_sum = torch.sum(r,dim=self.dim,keepdim=True)
        if torch.sum(r_sum==0):
            r_sum[r_sum==0] = 1
        robust = torch.div(r,r_sum)
        return robust


class NormRobust(object):
    def __init__(self, smax, scale):
        self.smax = smax
        self.scale = scale
    def forward(self, s, r, d):
        eps = 1e-12
        r_w = s*r
        mx = torch.abs(torch.max(r_w,dim=d,keepdim=True)[0])
        r_re = self.scale*torch.div(r_w,(mx+eps)) #rescale r
        # r_re = r
        s_norm = self.smax.forward(r_re) # weight of r_re
        return s_norm


class Disjunction(object):
    def __init__(self):
        pass
    def forward(self, X, w): # OR, EVENTUALLY
        s = torch.clone(X)
        w_sum = w.sum()
        if w_sum == 0:
            w_norm = w
        else:
            w_norm = w / w_sum
        s_norm = self.normalize_robust.forward(w_norm, s, 1)
        sw = torch.sum(torch.mul(s_norm,w_norm),dim=1)
        if torch.any(sw==0):
            s_norm[sw==0,:] = 0.1
        denominator = torch.mul(s_norm, w_norm)
        denominator = denominator
        denominator = torch.sum(denominator,dim=1,keepdim=True)
        
        numerator = torch.mul(s_norm, w_norm)
        numerator = torch.mul(numerator, s)
        numerator = torch.sum(numerator,dim=1,keepdim=True)
        denominator_old = torch.clone(denominator)
        denominator[(denominator_old==0)] = 1
        robust = numerator/denominator
        if torch.sum(denominator_old == 0): # there exists zero denominator
            if torch.all(denominator_old==0): # if all denominator equal zero
                robust[(denominator_old==0)] = -1
            else:
                robust[(denominator_old==0)] = torch.min(robust[(denominator_old!=0)])
        return robust

    def init_sparsemax(self, beta, a, scale, dim):
        self.smax = SparseMax(beta, a, dim)
        self.normalize_robust = NormRobust(self.smax, scale)


class Conjunction(object): # AND, ALWAYS
    def __init__(self):
        pass
    def forward(self, X, w): # OR, EVENTUALLY
        s = torch.clone(-X)
        w_sum = w.sum()
        if w_sum == 0:
            w_norm = w
        else:
            w_norm = w / w_sum
        s_norm = self.normalize_robust.forward(w_norm, s, 1)
        sw = torch.sum(torch.mul(s_norm,w_norm),dim=1)
        if torch.any(sw==0):
            s_norm[sw==0,:] = 0.1
        denominator = torch.mul(s_norm, w_norm)
        denominator = torch.sum(denominator,dim=1,keepdim=True)
        
        numerator = torch.mul(s_norm, w_norm)
        numerator = torch.mul(numerator, s)
        numerator = torch.sum(numerator,dim=1,keepdim=True)
        denominator_old = torch.clone(denominator)
        numerator_old = torch.clone(numerator)
        denominator[(denominator_old==0)] = 1
        robust = -numerator/denominator
        if torch.sum(denominator_old == 0): # there exists zero denominator
            if torch.all(denominator_old==0): # if all denominator equal zero
                robust[(denominator_old==0)] = -1
            else:
                robust[(denominator_old==0)] = torch.min(robust[(denominator_old!=0)])
        return robust

    def init_sparsemax(self, beta, a, scale, dim):
        self.smax = SparseMax(beta, a, dim)
        self.normalize_robust = NormRobust(self.smax, scale)



class Eventually(object):
    def __init__(self, A, b, t1, t2):
        self.A = A
        self.b = b
        self.t1 = t1
        self.t2 = t2
        self.duration = t2-t1+1

    def initialize_robustness(self, x):
        Ar = (self.A).repeat(self.batch_size,1,1)
        r = torch.matmul(Ar,x) - self.b
        r_pad = torch.ones((self.batch_size,1,self.duration), dtype=torch.float)*(-1)
        r_new = torch.cat((r,r_pad),-1)
        return r_new
    
    def robustness(self, X, w, t1, t2):
        r = X[:,:,t1:t2+1]
        s = torch.clone(r)
        w_sum = w.sum()
        if w_sum == 0:
            w_norm = w
        else:
            w_norm = w / w_sum
        s_norm = self.normalize_robust.forward(w_norm, s, 2)
        sw = torch.sum(torch.mul(s_norm,w_norm),dim=2)
        if torch.any(sw==0):
            s_norm[sw==0,:] = 0.1
        denominator = torch.mul(s_norm, w_norm)
        denominator = torch.sum(denominator,dim=2,keepdim=True)

        numerator = torch.mul(s_norm, w_norm)
        numerator = torch.mul(numerator, s)
        numerator = torch.sum(numerator,dim=2,keepdim=True)
        denominator_old = torch.clone(denominator)
        denominator[(denominator_old==0)] = 1
        robust = numerator/denominator
        if torch.sum(denominator_old == 0): # there exists zero denominator
            if torch.all(denominator_old==0): # if all denominator equal zero
                robust[(denominator_old==0)] = -1
            else:
                robust[(denominator_old==0)] = torch.min(robust[(denominator_old!=0)])
        return robust

    def robustness_trace(self, data, w, batch_size, need_trace):
        self.batch_size = batch_size
        self.data_dim = data.shape[1]
        X = self.initialize_robustness(data)
        if need_trace:
            trace = torch.empty((self.batch_size,self.data_dim,self.duration), dtype = torch.float)
            for i in range(self.duration):
                t1 = self.t1+i
                t2 = t1+self.duration-1
                trace[:,:,i] = self.robustness(X,w,t1,t2)
        else:
            t1 = self.t1
            t2 = self.t2
            trace = self.robustness(X,w,t1,t2)
        return trace

    def init_sparsemax(self, beta, a, scale, dim):
        self.smax = SparseMax(beta, a, dim)
        self.normalize_robust = NormRobust(self.smax, scale)
    


class Always(object):
    def __init__(self, A, b, t1, t2):
        self.A = A
        self.b = b
        self.t1 = t1
        self.t2 = t2
        self.duration = t2-t1+1
    
    def initialize_robustness(self, x):
        Ar = (self.A).repeat(self.batch_size,1,1)
        r = torch.matmul(Ar,x) - self.b
        r_pad = torch.ones((self.batch_size,1,self.duration), dtype=torch.float)*(-1)
        r_new = torch.cat((r,r_pad),-1)
        return r_new
    
    def robustness(self, X, w, t1, t2):
        r = X[:,:,t1:t2+1]
        s = torch.clone(-r)
        w_sum = w.sum()
        if w_sum == 0:
            w_norm = w
        else:
            w_norm = w / w_sum
        s_norm = self.normalize_robust.forward(w_norm, s, 2)
        sw = torch.sum(torch.mul(s_norm,w_norm),dim=2)
        if torch.any(sw==0):
            s_norm[sw==0,:] = 0.1
        denominator = torch.mul(s_norm, w_norm)
        denominator = torch.sum(denominator,dim=2,keepdim=True)

        numerator = torch.mul(s_norm, w_norm)
        numerator = torch.mul(numerator, s)
        numerator = torch.sum(numerator,dim=2,keepdim=True)
        denominator_old = torch.clone(denominator)
        denominator[(denominator_old==0)] = 1
        robust = -numerator/denominator
        if torch.sum(denominator_old == 0): # there exists zero denominator
            if torch.all(denominator_old==0): # if all denominator equal zero
                robust[(denominator_old==0)] = -1
            else:
                robust[(denominator_old==0)] = torch.min(robust[(denominator_old!=0)])
        return robust
    
    def robustness_trace(self, data, w, batch_size, need_trace):
        self.batch_size = batch_size
        self.data_dim = data.shape[1]
        X = self.initialize_robustness(data)
        if need_trace:
            trace = torch.empty((self.batch_size,self.data_dim,self.duration), dtype = torch.float)
            for i in range(self.duration):
                t1 = self.t1+i
                t2 = t1+self.duration-1
                trace[:,:,i] = self.robustness(X,w,t1,t2)
        else:
            t1 = self.t1
            t2 = self.t2
            trace = self.robustness(X,w,t1,t2)
        return trace

    def init_sparsemax(self, beta, a, scale, dim):
        self.smax = SparseMax(beta, a, dim)
        self.normalize_robust = NormRobust(self.smax, scale)