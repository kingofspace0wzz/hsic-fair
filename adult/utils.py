import torch
import torch.nn as nn
import torch.nn.functional as F

def batch_permutation(z, S):
    '''
    z : [N, D]
    s : list of protected factors [N, S]
    '''
    N = z.size(0)
    shuffle_idx = torch.randperm(N)
    z_perm = torch.stack([z[i] for i in shuffle_idx], dim=0)
    for k in range(len(S)):
        shuffle_idx = torch.randperm(N)
        s_perm = torch.stack([S[i] for i in shuffle_idx], dim=0)
        S[k] = s_perm
    zs_perm= torch.cat((z, torch.cat(S, dim=-1)), dim=-1)
    return z, S, zs_perm

def dot_kernel(z1, z2):
    return torch.matmul(z1.t(), z2)
