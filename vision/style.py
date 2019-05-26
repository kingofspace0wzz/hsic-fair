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
from dataloader import get_yale
from utils import batch_rotate
from loss import (recon_loss, total_kld, disc_loss, HSIC)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/mnist',
                    help='location of the data')
parser.add_argument('--batch_size', type=int, default=256),
parser.add_argument('--epochs', type=int, default=50),
parser.add_argument('--code_dim', type=int, default=20)
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
    elif dataset == 'fashion':
        train_loader, test_loader = get_fashion_mnist(args.batch_size, 'data/fashion')
    elif dataset == 'svhn':
        train_loader, test_loader, _ = get_svhn(args.batch_size, 'data/svhn')
    elif dataset == 'stl':
        train_loader, test_loader, _ = get_stl10(args.batch_size, 'data/stl10')
    elif dataset == 'cifar':
        train_loader, test_loader = get_cifar(args.batch_size, 'data/cifar')
    elif dataset == 'chair':
        train_loader, test_loader = get_chair(args.batch_size, '~/data/rendered_chairs')
    elif dataset == 'yale':
        train_loader, test_loader = get_yale(args.batch_size, 'data/yale')

    model = VAE(28*28, args.code_dim, args.batch_size, 10, dataset).to(device)
    phi = nn.Sequential(
        nn.Linear(args.code_dim, args.phi_dim),
        nn.LeakyReLU(0.2, True),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_phi = torch.optim.Adam(phi.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='sum')
    for epoch in range(args.epochs):
        re_loss = 0
        kl_div = 0
        size = len(train_loader.dataset)
        for data, target in train_loader:
            data, target = data.squeeze(1).to(device), target.to(device)
            c = F.one_hot(target.long(), num_classes=10).float()
            output, q_z, p_z, z = model(data, c)
            hsic = HSIC(phi(z), target.long())
            if dataset == 'mnist' or dataset == 'fashion':
                reloss = recon_loss(output, data.view(-1, 28*28))
            else:
                reloss = criterion(output, data)
            kld = total_kld(q_z, p_z)
            loss = reloss + kld + args.c * hsic
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_phi.zero_grad()
            neg = -HSIC(phi(z.detach()), target.long())
            neg.backward()
            optimizer_phi.step()

            re_loss += reloss.item() / size
            kl_div += kld.item() / size
        print('-'*50)
        print(" Epoch {} |re loss {:5.2f} | kl div {:5.2f} | hs {:5.2f}".format(epoch, re_loss, kl_div, hsic))
    for data, target in test_loader:
        data, target = data.squeeze(1).to(device), target.to(device)
        c = F.one_hot(target.long(), num_classes=10).float()
        output, _, _, z = model(data, c)
        break
    if dataset == 'mnist' or dataset == 'fashion':
        img_size = [data.size(0), 1, 28, 28]
    else:
        img_size = [data.size(0), 3, 32, 32]
    images = [data.view(img_size)[:30].cpu()]
    for i in range(10):
        c = F.one_hot(torch.ones(z.size(0)).long()*i, num_classes=10).float().to(device)
        output = model.decoder(torch.cat((z, c), dim=-1))
        images.append(output.view(img_size)[:30].cpu())
    images = torch.cat(images, dim=0)
    save_image(images, 'imgs/recon_c{}_{}.png'.format(int(args.c), dataset), nrow=30)
    torch.save(model.state_dict(), 'vae_c{}_{}.pt'.format(int(args.c), dataset))
    # z = p_z.sample()
    # for i in range(10):
    #     c = F.one_hot(torch.ones(z.size(0)).long()*i, num_classes=10).float().to(device)
    #     output = model.decoder(torch.cat((z, c), dim=-1))
    #     n = min(z.size(0), 8)
    #     save_image(output.view(z.size(0), 1, 28, 28)[:n].cpu(), 'imgs/recon_{}.png'.format(i), nrow=n)
    if args.tsne:
        data, target = test_loader.dataset[:250]
        z = model.encoder(data.to(args.device))
        z, target = z.detach().cpu().numpy(), target.cpu().numpy()
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        z_2d = tsne.fit_transform(z)
        plt.figure(figsize=(6, 5))
        plot_embedding(z_2d, target)
        plt.savefig('tsne_c{}_{}.png'.format((int(args.c), dataset)))

def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1],
                 c=colors[int(y[i])],
                 label=int(y[i]))
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

if __name__ == "__main__":
    main(args)