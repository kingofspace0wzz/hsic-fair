import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import numpy as np

def recon_loss(recon, target):
    loss = F.binary_cross_entropy(recon, target, reduction='sum')
    return loss 

def total_kld(q_z, p_z):
    return torch.sum(kl_divergence(q_z, p_z))

def disc_loss(D_z, target):
    loss = F.cross_entropy(D_z, target, reduction='sum')
    return loss

def perm_loss(joint, indepedent):
    zeros = torch.zeros(joint.size(0), dtype=torch.long, device=joint.device)
    ones = torch.ones(indepedent.size(0), dtype=torch.long, device=indepedent.device)
    loss = 0.5 * (F.cross_entropy(joint, zeros) + F.cross_entropy(indepedent, ones))
    return loss
    
def HSIC(z, s, fix=False):
    n = z.size(0)
    # k(x, y) = <z(x), z(y)>, K = z * z^T
    if fix:
        K = rbf(z, z)
    else:
        K = torch.matmul(z, z.t())
    # H = I - 1 * 1^T / (n-1)^2
    H = torch.eye(z.size(0)).to(z.device) - torch.ones_like(K) / n
    
    # encode protected factor into one_hot
    h = F.one_hot(s).float()
    # L = h * h^T
    L = torch.matmul(h, h.t())
    return torch.sum(torch.diag(torch.chain_matmul(K,H,L,H))) / (n-1)**2

def rbf(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-0.5*kernel)
