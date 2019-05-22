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

def kernel_loss(weight1, weight2, z, kernel):
    loss = torch.pow(torch.norm(torch.matmul(weight1, torch.matmul(kernel(F.relu(z), F.relu(z)), weight2.t()))), 2)
    # loss = torch.pow(torch.norm(torch.matmul(weight1, weight2.t())), 2)
    return loss

def HSIC(z, s):
    n = z.size(0)
    # k(x, y) = <z(x), z(y)>, K = z * z^T
    K = torch.matmul(z, z.t())
    # H = I - 1 * 1^T / (n-1)^2
    H = torch.eye(z.size(0)).to(z.device) - torch.ones_like(K) / n
    
    # encode protected factor into one_hot
    h = F.one_hot(s).float()
    # L = h * h^T
    L = torch.matmul(h, h.t())
    return torch.sum(torch.diag(torch.chain_matmul(K,H,L,H))) / (n-1)**2
    
def COCO(z, s):
    n = z.size(0)
    K = torch.matmul(z, z.t())
    H = torch.eye(n).to(z.device) - torch.ones_like(K) / n

    

def KCC(z, s):
    pass

def KMI(z, s):
    pass

def centering(K):
    n = K.size(0)
    unit = torch.ones_like(K)
    I = torch.eye(n).to(K.device)
    Q = I - unit / n
    return torch.matmul(torch.matmul(Q, K), Q)

class hinge_loss(nn.Module):
    def __init__(self):
        super(hinge_loss, self).__init__()

    def forward(self, output, target):
        return torch.clamp(1 - torch.mul(output, target.float()), min=0)

