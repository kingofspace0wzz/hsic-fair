import argparse
import time
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import pyro
import pyro.contrib.gp as gp
import pyro.infer as infer

from data import get_adult
from model import Encoder, Classifier, Model
from utils import batch_permutation, dot_kernel
from loss import perm_loss, kernel_loss, hinge_loss

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='adult.data')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--code_dim', type=int, default=32)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--drop', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--prepochs', type=int, default=20)
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--c', type=float, default=10)
parser.add_argument('--nogender', action='store_true', help="drop gender info")
parser.add_argument('--base', action='store_true', help="use baseline")
parser.add_argument('--adv', action='store_true', help="use Controllable invariance with adversarial training")
parser.add_argument('--loss', type=str, default='entropy')
parser.add_argument('--jit', action='store_true')
parser.add_argument('--num-inducing', type=int, default=70)
parser.add_argument('--seed', type=int, default=22)
parser.add_argument('--test_size', type=float, default=0.5)
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

def train(args, train_loader, gpmodule, optimizer, criterion, epoch):
    batch_size = args.batch_size
    device = args.device
    clf_loss = 0
    clf_acc = 0
    correct = 0
    correct_r = 0
    correct_g = 0
    total = len(train_loader.dataset)
    total_m = 0
    total_f = 0
    y_m = 0
    y_f = 0
    for i, (data, target, factors) in enumerate(train_loader):
        data, target = data.to(device), target.long().squeeze(1).to(device)
        label_r = factors.chunk(2, dim=-1)[0].squeeze(1).long()
        label_g = factors.chunk(2, dim=-1)[1].squeeze(1).long()
        gpmodule.set_data(data, target)
        optimizer.zero_grad()
        loss = criterion(gpmodule.model, gpmodule.guide)
        loss.backward()
        optimizer.step()

        clf_loss += loss.item()

        f_loc, f_var = gpmodule(data)
        pred = gpmodule.likelihood(f_loc, f_var)
        correct += pred.eq(target).long().cpu().sum()

    clf_acc = 100 * correct / total

    return clf_loss, clf_acc

def test(args, test_loader, gpmodule):
    device = args.device
    correct = 0
    total = len(test_loader.dataset)
    for i, (data, target, factors) in enumerate(test_loader):
        data, target = data.to(device), target.long().squeeze(1).to(device)
        label_r = factors.chunk(2, dim=-1)[0].squeeze(1).long()
        label_g = factors.chunk(2, dim=-1)[1].squeeze(1).long()
        f_loc, f_var = gpmodule(data)
        pred = gpmodule.likelihood(f_loc, f_var)
        correct += pred.eq(target).long().cpu().sum()
    
    clf_acc = 100 * correct / total
    return clf_acc

def main(args):
    torch.cuda.set_device(args.cuda)
    print("Loading data")
    dataset = args.data.rstrip('/').split('/')[-1]
    train_iter, test_iter = get_adult(args.data, args.batch_size, args.nogender, args.test_size)
    
    for _, (batch, _, _) in enumerate(train_iter):
        n_features = batch.size(-1)
        break
    encoder = Classifier(n_features, args.code_dim, args.hidden)
    rbf = gp.kernels.RBF(input_dim=args.code_dim, lengthscale=torch.ones(args.code_dim))
    deep_kernel = gp.kernels.Warping(rbf, iwarping_fn=encoder)

    batches = []
    for i, (data, _, _) in enumerate(train_iter):
        batches.append(data)
        # if i >= ((args.num_inducing - 1) // args.batch_size):
            # break
    X = torch.cat(batches)[:].clone()
    Xu = torch.cat(batches)[:args.num_inducing].clone()
    
    likelihood = gp.likelihoods.MultiClass(num_classes=2)
    latent_shape = torch.Size([2])

    gpmodule = gp.models.VariationalSparseGP(X=X, y=None, kernel=deep_kernel, Xu=Xu,
                                        likelihood=likelihood, latent_shape=latent_shape,
                                        whiten=True).to(args.device)

    optimizer = torch.optim.Adam(gpmodule.parameters(), lr=args.lr)
    elbo = infer.JitTraceMeanField_ELBO if args.jit else infer.TraceMeanField_ELBO()
    criterion = elbo.differentiable_loss

    start_epoch = 1
    print('\nStart training')
    try:
        for epoch in range(start_epoch, args.epochs):
            clf_loss, clf_acc = train(args, train_iter, gpmodule, optimizer, criterion, epoch)
            print('-' * 90)
            meta = "| epoch {:2d} ".format(epoch)
            print(meta + "| Train loss {:5.2f} | Train acc {:5.2f}".format(clf_loss, clf_acc))

            clf_acc = test(args, test_iter, gpmodule)
            print(len(meta)* " " + "| Test loss {:5.2f} | Test accuracy {:5.2f}".format(clf_loss, clf_acc))

    except KeyboardInterrupt:
        print('-'*50)
        print('Quit training')

if __name__ == "__main__":
    main(args)