from __future__ import print_function
import torch
import torch.utils.data

import math

MIN_EPSILON = 1e-5
MAX_EPSILON = 1. - 1e-5

PI = torch.FloatTensor([math.pi])
EPS=torch.FloatTensor([1e-10])
if torch.cuda.is_available():
    PI = PI.cuda()
    EPS=EPS.cuda()



def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_normalized(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -(x - mean) * (x - mean)
    log_norm *= torch.reciprocal(2. * log_var.exp())
    log_norm += -0.5 * log_var
    log_norm += -0.5 * torch.log(2. * PI)

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_bernoulli(x, mean, average=False, reduce=True, dim=None):
    probs = torch.clamp(mean, min=MIN_EPSILON, max=MAX_EPSILON)
    log_bern = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    if reduce:
        if average:
            return torch.mean(log_bern, dim)
        else:
            return torch.sum(log_bern, dim)
    else:
        return log_bern

class TwoNomal():
    def __init__(self,mu1,mu2,var1,var2):
        self.mu1 = mu1
        self.var1 = var1
        self.mu2 = mu2
        self.var2 = var2
    def log_doubledensity(self,x):
            mu1 = self.mu1
            var1 = self.var1
            mu2 = self.mu2
            var2 = self.var2
            N1 = torch.sqrt(2 * PI * var1)
            fac1 = (x - mu1)** 2 / var1
            density1=torch.exp(-fac1/2)/N1

            N2 = torch.sqrt(2 * PI * var2)
            fac2 = (x - mu2)**2 / var2
            density2=torch.exp(-fac2/2)/N2
            density=0.5*density2+0.5*density1
            return torch.log(density+EPS)