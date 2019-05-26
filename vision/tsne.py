import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import save_image

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

from model import VAE, Discriminator, YaleB, YaleBHSIC
from data import get_mnist, get_cifar, get_fashion_mnist, get_stl10, get_svhn, get_chair, get_yale
# from dataloader import get_yale
from utils import batch_rotate
from loss import (recon_loss, total_kld, disc_loss, HSIC)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/mnist',
                    help='location of the data')
parser.add_argument('--batch_size', type=int, default=256),
parser.add_argument('--epochs', type=int, default=50),
parser.add_argument('--code_dim', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3),
parser.add_argument('--wd', type=float, default=0,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--kla', action='store_true'),
parser.add_argument('--save_name', type=str, default="trained_model.pt")
parser.add_argument('--loss_name', type=str, default="logger.pt")
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--ro', action='store_true')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--base', action='store_true')
parser.add_argument('--D', action='store_true')
parser.add_argument('--c', type=float, default=10)
parser.add_argument('--lepochs', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.6)
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--phi_dim', type=int, default=128)
parser.add_argument('--fix', action='store_true')
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--fname', type=str)
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

def main(args):
    print("Loading data")
    dataset = args.data.rstrip('/').split('/')[-1]
    torch.cuda.set_device(args.cuda)
    device = args.device
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist(args.batch_size, 'data/mnist')
        num = 10
    elif dataset == 'fashion':
        train_loader, test_loader = get_fashion_mnist(args.batch_size, 'data/fashion')
        num = 10
    elif dataset == 'svhn':
        train_loader, test_loader, _ = get_svhn(args.batch_size, 'data/svhn')
        num = 10
    elif dataset == 'stl':
        train_loader, test_loader, _ = get_stl10(args.batch_size, 'data/stl10')
    elif dataset == 'cifar':
        train_loader, test_loader = get_cifar(args.batch_size, 'data/cifar')
        num = 10
    elif dataset == 'chair':
        train_loader, test_loader = get_chair(args.batch_size, '~/data/rendered_chairs')
        num = 1393
    elif dataset == 'yale':
        train_loader, test_loader = get_yale(args.batch_size, 'data/yale')
        num = 38
    model = VAE(28*28, args.code_dim, args.batch_size, 10, dataset).to(device)
    phi = nn.Sequential(
        nn.Linear(args.code_dim, args.phi_dim),
        nn.LeakyReLU(0.2, True),
    ).to(device)
    model.load_state_dict(torch.load(args.fname))
    if args.tsne:
        datas, targets = [], []
        for i, (data, target) in enumerate(test_loader):
            datas.append(data), targets.append(target)
            if i >= 5:
                break
        data, target = torch.cat(datas, dim=0), torch.cat(targets, dim=0)
        c = F.one_hot(target.long(), num_classes=num).float()
        _, _, _, z = model(data.to(args.device), c.to(args.device))
        z, target = z.detach().cpu().numpy(), target.cpu().numpy()
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        z_2d = tsne.fit_transform(z)
        plt.figure(figsize=(6, 5))
        for a in range(8):
            for b in range(a+1, 10):
                plot_embedding(z_2d, target, a, b, )
                plt.savefig('tsne_c{}_{}_{}{}.png'.format(int(args.c), dataset, a, b))

def plot_embedding(X, y, a, b, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'lightpink', 'orange', 'purple'
    # colors = 'w', 'w', 'b', 'w', 'w', 'w', 'w', 'w', 'w', 'purple'
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        if int(y[i]) == a or int(y[i]) == b:
            plt.scatter(X[i, 0], X[i, 1],
                    c=colors[int(y[i])],
                    label=int(y[i]))
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

if __name__ == "__main__":
    main(args)