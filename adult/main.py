import argparse
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.contrib.gp as gp

from data import get_adult, get_german
from model import Encoder, Classifier, Model
from utils import batch_permutation, dot_kernel
from loss import perm_loss, kernel_loss, hinge_loss, HSIC

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
parser.add_argument('--gepochs', type=int, default=10)
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--c', type=float, default=10)
parser.add_argument('--nogender', action='store_true', help="drop gender info")
parser.add_argument('--base', action='store_true', help="use baseline")
parser.add_argument('--adv', action='store_true', help="use Controllable invariance with adversarial training")
parser.add_argument('--loss', type=str, default='entropy')
parser.add_argument('--test_size', type=float, default=0.5)
parser.add_argument('--ge', action='store_true')
parser.add_argument('--hsic', action='store_true')
parser.add_argument('--seed', type=int, default=22)
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

def run(args, data_iter, model, gender, optimizers, epoch, train=True, pretrain=False):
    n = args.batch_size
    size = len(data_iter.dataset)
    device = args.device
    dataset = args.data.rstrip('/').split('/')[-1]
    if args.loss == 'hinge':
        criterion = hinge_loss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer, optimizer_phi = optimizers
    # kernel = gp.kernels.RBF(input_dim=args.code_dim*3, variance=torch.tensor(5.),
                            # lengthscale=torch.tensor(10.))
    # kernel = gp.kernels.Linear(input_dim=args.code_dim)
    kernel = dot_kernel
    if train:
        model.train()
    else:
        model.eval()

    clf_loss = 0
    clf_acc = 0
    correct = 0
    correct_g = 0
    correct_ge = 0
    total = 0
    total_m = 0
    total_f = 0
    y_m = 0
    y_f = 0
    hs = 0
    for i, data in enumerate(data_iter):
        inputs, label, factor = [d.to(device) for d in data] 
        label = label.long().squeeze(1)
        if dataset == 'adult.data':
            label_g = factor.chunk(2, dim=-1)[1].squeeze(1).long()
        else:
            label_g = factor.long().squeeze(1)
            # _, label_g = torch.max(label_g, 1)
        # label = label.long()
        # label_g = factor.chunk(2, dim=-1)[0].long()
        # label_r = factor.chunk(2, dim=-1)[0].long()

        y, z, _ = model(inputs)
        phi = model.classifier.map(F.relu(z))
        loss = criterion(y, label)
        hsic = HSIC(phi, label_g)
        total_loss = loss + args.c * hsic
        
        if train:
                
            if args.hsic:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                optimizer_phi.zero_grad()
                phi = model.classifier.map(F.relu(z.detach()))
                neg_h = -HSIC(phi, label_g)
                neg_h.backward()
                optimizer_phi.step()
        
        _, predicted = torch.max(y.data, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)
        
        ones = torch.ones(label.size(0), dtype=torch.long).to(device)
        zeros = torch.zeros(label.size(0), dtype=torch.long).to(device)
        total_m += (ones == label_g).sum().item()
        total_f += (zeros == label_g).sum().item()
        y_m += ((predicted == ones) == (ones == label_g)).sum().item()
        y_f += ((predicted == ones) == (zeros == label_g)).sum().item()

        # predicted = (torch.sigmoid(y) >= 0.5).long()
        # predicted_r = (torch.sigmoid(r) >= 0.5).long()
        # predicted_g = (torch.sigmoid(g) >= 0.5).long()
        # correct += (predicted == label).sum().item()
        # correct_r += (predicted_r == label_r).sum().item()
        # correct_g += (predicted_g == label_g).sum().item()
        # total += label.size(0)
        # ones = torch.ones((label.size(0), 1), dtype=torch.long).to(device)
        # zeros = torch.zeros((label.size(0), 1), dtype=torch.long).to(device)
        # total_m += (ones == label_g).sum().item(), optimizer_ge
        # total_f += (zeros == label_g).sum().item()
        # y_m += ((predicted == ones) == (ones == label_g)).sum().item()
        # y_f += ((predicted == zeros) == (zeros == label_g)).sum().item()
        # print(((predicted == ones) == (zeros == label_g)).sum().item())

        clf_loss += loss.item()
        hs += hsic.item()
    
    clf_acc = 100 * correct / total
    parity = np.abs(y_m / total_m - y_f / total_f)
    male = total_m / total

    return clf_loss, clf_acc, parity, hs  

def baseline(args, data_iter, model, optimizers, epoch, train=True):
    n = args.batch_size
    size = len(data_iter.dataset)
    device = args.device
    criterion = torch.nn.CrossEntropyLoss()
    optimizer, optimizer_r, optimizer_g = optimizers
    if train:
        model.train()
    else:
        model.eval()

    clf_loss = 0
    clf_acc = 0
    correct = 0
    correct_r = 0
    correct_g = 0
    total = 0
    total_m = 0
    total_f = 0
    y_m = 0
    y_f = 0
    for i, data in enumerate(data_iter):
        inputs, label, factor = [d.to(device) for d in data] 
        label = label.long().squeeze(1)
        label_r = factor.chunk(2, dim=-1)[0].squeeze(1).long()
        label_g = factor.chunk(2, dim=-1)[1].squeeze(1).long()
        y, z = model(inputs)
        r, g = model.race(F.relu(z.detach())), model.gender(F.relu(z.detach()))
        loss = criterion(y, label)
        loss_r = criterion(r, label_r)
        loss_g = criterion(g, label_g)
        if args.loss == 'hinge':
            loss = torch.mean(torch.clamp(1 - torch.mul(y, label.float()), min=0))
            loss_g = torch.mean(torch.clamp(1 - torch.mul(g, label_g.float()), min=0))
            loss_r = torch.mean(torch.clamp(1 - torch.mul(r, label_r.float()), min=0)) 
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_r.zero_grad()
            loss_r.backward()
            optimizer_r.step()

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
        
        _, predicted = torch.max(y.data, 1)
        _, predicted_r = torch.max(r.data, 1)
        _, predicted_g = torch.max(g.data, 1)
        correct += (predicted == label).sum().item()
        correct_r += (predicted_r == label_r).sum().item()
        correct_g += (predicted_g == label_g).sum().item()
        total += label.size(0)

        ones = torch.ones(label.size(0), dtype=torch.long).to(device)
        zeros = torch.zeros(label.size(0), dtype=torch.long).to(device)
        total_m += (ones == label_g).sum().item()
        total_f += (zeros == label_g).sum().item()
        y_m += ((predicted == ones) == (ones == label_g)).sum().item()
        y_f += ((predicted == zeros) == (zeros == label_g)).sum().item()
    
        clf_loss += loss.item()
    
    clf_acc = 100 * correct / total
    r_acc = 100 * correct_r / total
    g_acc = 100 * correct_g / total
    parity = np.abs(y_m / total_m - y_f / total_f)

    return clf_loss, clf_acc, r_acc, g_acc, parity

def main(args):
    torch.cuda.set_device(args.cuda)
    print("Loading data")
    dataset = args.data.rstrip('/').split('/')[-1]
    if dataset == 'adult.data':
        train_iter, test_iter = get_adult(args.data, args.batch_size, args.nogender, args.test_size)
    else:
        train_iter, test_iter = get_german(args.data, args.batch_size, args.test_size)

    for _, (batch, _, _) in enumerate(train_iter):
        n_features = batch.size(-1)
        break
    if args.loss == 'hinge':
        model = Model(n_features, args.code_dim, args.hidden, 1, 2, args.drop).to(args.device)
    else:
        model = Model(n_features, args.code_dim, args.hidden, 2, 2, args.drop).to(args.device)
    # race = Classifier(args.code_dim*3, 2, args.hidden).to(args.device)
    # gender = Classifier(args.code_dim*3, 2, args.hidden).to(args.device)
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + 
                                list(model.classifier.net.parameters()), lr=args.lr)
    optimizer_phi = torch.optim.Adam(list(model.classifier.map.parameters()), lr=args.lr)
    # gender = Classifier(args.code_dim, 2, args.hidden).to(args.device)
    gender = nn.Linear(args.code_dim, 2).to(args.device)
    optimizer_ge = torch.optim.Adam(gender.parameters(), lr=args.lr)
    start_epoch = 1
    print('\nStart training')
    if args.base:
        try:
            for epoch in range(start_epoch, args.epochs):
                clf_loss, clf_acc, r_acc, g_acc, _ = baseline(args, train_iter, model, (optimizer, optimizer_r, optimizer_g), epoch, True)
                print('-' * 90)
                meta = "| epoch {:2d} ".format(epoch)
                print(meta + "| Train loss {:5.2f} | Train acc {:5.2f} | Gender acc {:5.2f} | Race acc {:5.2f}".format(clf_loss, clf_acc, g_acc, r_acc))

                clf_loss, clf_acc, r_acc, g_acc, parity = baseline(args, test_iter, model, (optimizer, optimizer_r, optimizer_g), epoch, False)
                print(len(meta)* " " + "| Test loss {:5.2f} | Test accuracy {:5.2f} | Gender acc {:5.2f} | Race acc {:5.2f} | Parity {:5.2f}".format(clf_loss, clf_acc, g_acc, r_acc, parity))

        except KeyboardInterrupt:
            print('-'*50)
            print('Quit training')
    else:
        try:
            if args.pretrain:
                for epoch in range(start_epoch, args.prepochs):
                    clf_loss, clf_acc, parity, hs = run(args, train_iter, model, gender, (optimizer, optimizer_phi), epoch, True, True)
                    print('-' * 90)
                    meta = "| epoch {:2d} ".format(epoch)
                    print(meta + "| pre Train loss {:5.2f} | pre Train acc {:5.2f} | Gender acc {:5.2f}".format(clf_loss, clf_acc, g_acc))

                    clf_loss, clf_acc, parity, hs = run(args, test_iter, model, gender, (optimizer, optimizer_phi), epoch, False, True)
                    print(len(meta)* " " + "| Test loss {:5.2f} | Test accuracy {:5.2f} | Gender acc {:5.2f} | Parity {:5.2f} | adv {:5.2f} | hs {:5.2f}".format(clf_loss, clf_acc, g_acc, parity, adv_acc, hs))
                    print('-'*50)
                print('End pretrain')
            for epoch in range(start_epoch, args.epochs):
                clf_loss, clf_acc, parity, hs = run(args, train_iter, model, gender, (optimizer, optimizer_phi), epoch, True, True)
                print('-' * 90)
                meta = "| epoch {:2d} ".format(epoch)
                print(meta + "| Train loss {:5.2f} | Train acc {:5.2f} | ".format(clf_loss, clf_acc))

                clf_loss, clf_acc, parity, hs = run(args, train_iter, model, gender, (optimizer, optimizer_phi), epoch, False, True)
                print(len(meta)* " " + "| Test loss {:5.2f} | Test accuracy {:5.2f} | Parity {:5.2f} | hs {:5.2f}".format(clf_loss, clf_acc, parity, hs))
        
        except KeyboardInterrupt:
            print('-'*50)
            print('Quit training')    

    if args.ge:
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(args.gepochs):
            gender.train()
            correct = 0
            total = 0
            total_m = 0
            hs = 0
            for data in train_iter:
                inputs, label, factor = [d.to(args.device) for d in data] 
                label = label.long().squeeze(1)
                if dataset == 'adult.data':
                    label_g = factor.chunk(2, dim=-1)[1].squeeze(1).long()
                else:
                    label_g = factor.squeeze(1).long()
                z = model.encoder(inputs)
                phi = model.classifier.map(z)
                ge = gender(F.relu(z.detach()))
                loss = loss_fn(ge, label_g)
                hs += HSIC(phi, label_g).item()
                optimizer_ge.zero_grad()
                loss.backward()
                optimizer_ge.step()
                _, predicted = torch.max(ge.data, 1)
                correct += (predicted == label_g).sum().item()
                total += label.size(0)
            adv_acc = 100 * correct / total
            print('-'*50)
            print('adv: {:5.2f} | hs {:5.2f}'.format(adv_acc, hs))
            gender.eval()
            correct = 0
            total = 0
            hs = 0
            for data in test_iter:
                inputs, label, factor = [d.to(args.device) for d in data] 
                label = label.long().squeeze(1)
                if dataset == 'adult.data':
                    label_g = factor.chunk(2, dim=-1)[1].squeeze(1).long()
                else:
                    label_g = factor.squeeze(1).long()
                z = model.encoder(inputs)
                phi = model.classifier.map(z)
                ge = gender(F.relu(z.detach()))
                _, predicted = torch.max(ge.data, 1)
                correct += (predicted == label_g).sum().item()
                total += label.size(0)
                hs += HSIC(phi, label_g).item()
                ones = torch.ones(label.size(0), dtype=torch.long).to(args.device)
                total_m += (ones == label_g).sum().item()
            male = total_m / total
            adv_acc = 100 * correct / total
            print('adv: {:5.2f} | hs {:5.2f} | m {:5.2f}'.format(adv_acc, hs, male))

if __name__ == "__main__":
    main(args)