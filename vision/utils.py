import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms

def toimg(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor)

def totensor(img):
    transform = transforms.ToTensor()
    return transform(img)

def batch_rotate(batch_img, angles):
    '''
    rotate a batch of images by random angles
    '''
    batch_size = batch_img.size(0)
    for i in range(batch_size):
        batch_img[i] = rotate(batch_img[i], angles[i].item())

    return batch_img

def rotate(img, angle):
    if angle == 0: # do not rotate
        return img
    elif angle == 1: # rotate by 90
        return img.flip(-1).transpose(-2, -1)
    elif angle == 2: # rotate by 180 
        return img.flip(-1).transpose(-2, -1).flip(-1).transpose(-2, -1)
    elif angle == 3: # rotate by 270
        return img.flip(-2).transpose(-2, -1)
    else:
        raise NotImplementedError

def batch_con_rotate(batch, angles):
    for i in range(batch.size(0)):
        img = TF.rotate(toimg(batch[i]), angles[i])
        batch[i] = totensor(img)
    return batch
        
def batch_rescale(batch, sizes):
    transform = [transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(s),
        transforms.ToTensor()
    ]) for s in sizes]
    for i in range(batch.size(0)):
        batch[i] = transform[i](batch[i])

    return batch

def batch_gamma(batch, gammas):
    for i in range(batch.size(0)):
        img = TF.adjust_gamma(toimg(batch[i]), gammas[i])
        batch[i] = totensor(img)
    return batch

def batch_hue(batch, hues):
    for i in range(batch.size(0)):
        img = TF.adjust_hue(toimg(batch[i]), hues[i])
        batch[i] = totensor(img)
    return batch

def batch_brightness(batch, b):
    for i in range(batch.size(0)):
        img = TF.adjust_brightness(toimg(batch[i]), b[i])
        batch[i] = totensor(img)
    return batch

def batch_satuation(batch, factor):
    for i in range(batch.size(0)):
        img = TF.adjust_saturation(toimg(batch[i]), factor[i])
        batch[i] = totensor(img)
    return batch

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