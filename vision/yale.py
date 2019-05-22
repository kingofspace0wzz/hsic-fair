import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from model import VAE, Discriminator, YaleB, YaleBHSIC
from data import get_mnist, get_cifar, get_fashion_mnist, get_stl10, get_svhn
from dataloader import get_yale
from utils import batch_rotate
from loss import (recon_loss, total_kld, disc_loss, HSIC)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/yale',
                    help='location of the data')
parser.add_argument('--batch_size', type=int, default=5),
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
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--base', action='store_true')
parser.add_argument('--D', action='store_true')
parser.add_argument('--c', type=float, default=10)
parser.add_argument('--lepochs', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.6)
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--phi_dim', type=int, default=128)
parser.add_argument('--fix', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

def run(args, data_iter, model, clf, optimizers, epoch, train=True, pretrain=False):
    batch_size = args.batch_size
    size = len(data_iter.dataset)
    device = args.device
    optimizer, optimizer_phi = optimizers
    if train:
        model.train()
    else:
        model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    # data_iter.init_epoch()
    clf_loss = 0
    hs = 0
    correct = 0
    correct_l = 0
    total = 0
    for i, (data, label, light) in enumerate(data_iter):
        data, label, light = data[:, 0, :, :].to(device), label.to(device), light.to(device)
        y, z = model(data.view(-1, 32*32))
        phi = model.phi(z)
        l = clf(F.relu(z.detach()))
        loss = criterion(y, label)
        hsic = HSIC(phi, light)
        total_loss = loss + args.c * HSIC(phi, light)

        if pretrain:
            optimizer_phi.zero_grad()
            phi = model.phi(z.detach())
            neg_h = -HSIC(phi, light)
            neg_h.backward()
            optimizer_phi.step()

        if train:
            optimizer_phi.zero_grad()
            # phi = model.phi(z.detach())
            # neg_h = -HSIC(phi, light)
            neg_h = -total_loss
            neg_h.backward(retain_graph=True)
            optimizer_phi.step()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


        clf_loss += loss.item()
        hs += hsic.item()
        total += data.size(0)
        _, predicted = torch.max(y.data, 1)
        _, predicted_l = torch.max(l.data, 1)
        correct += (predicted == label).sum().item()
        correct_l += (predicted_l == light).sum().item()
    acc = 100 * correct / total
    acc_l = 100 * correct_l / total
    
    return clf_loss, acc, acc_l, hs

def fix(args, data_iter, model, clf, optimizers, epoch, train=True, pretrain=False):
    batch_size = args.batch_size
    size = len(data_iter.dataset)
    device = args.device
    optimizer, optimizer_phi = optimizers
    if train:
        model.train()
    else:
        model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    # data_iter.init_epoch()
    clf_loss = 0
    hs = 0
    correct = 0
    correct_l = 0
    total = 0
    for i, (data, label, light) in enumerate(data_iter):
        data, label, light = data[:, 0, :, :].to(device), label.to(device), light.to(device)
        y, z = model(data.view(-1, 32*32))
        loss = criterion(y, label)
        l = clf(F.relu(z.detach()))
        hsic = HSIC(z, light, True)
        total_loss = loss + args.c * hsic

        if train:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        clf_loss += loss.item()
        hs += hsic.item()
        total += data.size(0)
        _, predicted = torch.max(y.data, 1)
        _, predicted_l = torch.max(l.data, 1)
        correct += (predicted == label).sum().item()
        correct_l += (predicted_l == light).sum().item()
    acc = 100 * correct / total
    acc_l = 100 * correct_l / total
    
    return clf_loss, acc, acc_l, hs

def baseline(args, data_iter, model, clf, optimizers, epoch, train=True):
    batch_size = args.batch_size
    size = len(data_iter.dataset)
    device = args.device
    optimizer, optimizer_c = optimizers
    if train:
        model.train()
    else:
        model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    # data_iter.init_epoch()
    clf_loss = 0
    correct = 0
    correct_l = 0
    total = 0
    for i, (data, label, light) in enumerate(data_iter):
        data, label, light = data[:, 0, :, :].to(device), label.to(device), light.to(device)
        y = model(data.view(-1, 32*32))
        l = clf(data.view(-1, 32*32))
        loss = criterion(y, label)
        loss_c = criterion(l, light)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

        clf_loss += loss.item()
        total += data.size(0)
        _, predicted = torch.max(y.data, 1)
        _, predicted_l = torch.max(l.data, 1)
        correct += (predicted == label).sum().item()
        correct_l += (predicted_l == light).sum().item()
    acc = 100 * correct / total
    acc_l = 100 * correct_l / total
    
    return clf_loss, acc, acc_l

def main(args):
    print("Loading data")
    dataset = args.data.rstrip('/').split('/')[-1]
    train_loader, test_loader = get_yale(args.batch_size, 'data/yale')
    torch.cuda.set_device(args.device_id)
    torch.autograd.set_detect_anomaly(True)
    if args.base:
        model = YaleB().to(args.device)
        clf = nn.Linear(192*168, 5).to(args.device)
    else:
        model = YaleBHSIC(args.phi_dim).to(args.device) 
        clf = nn.Linear(256, 5).to(args.device)
        clf2 = nn.Linear(256, 5).to(args.device)
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + 
                                 list(model.classifier.parameters()), lr=args.lr, weight_decay=args.wd)
    optimizer_c = torch.optim.Adam(clf.parameters(), lr=args.lr)
    optimizer_phi = torch.optim.Adam(model.phi.parameters(), lr=args.lr)

    start_epoch = 1
    print('\nStarting Training')
    if args.base:
        try:
            scheduler = MultiStepLR(optimizer, milestones=[50,100, 150], gamma=args.gamma)
            for epoch in range(start_epoch, args.epochs): 
                clf_loss, acc, acc_l = baseline(args, train_loader, model, clf, (optimizer, optimizer_c), epoch, train=True)
                scheduler.step()
                print('-' * 90)
                meta = "| epoch {:2d} ".format(epoch)
                print(meta + "| Train loss {:5.2f} | Train acc {:5.2f} | L acc {:5.2f}".format(clf_loss, acc, acc_l))

        except KeyboardInterrupt:
            print('-'*50)
            print('Quit Training')

        test_loss, test_acc, test_acc_l = baseline(args, test_loader, model, clf, (optimizer, optimizer_c), epoch, False)
        print('-'*50)
        print("| Test loss {:5.2f} | Test acc {:5.2f} | L acc {:5.2f}".format(test_loss, test_acc, test_acc_l))
    elif args.fix:
        try:
            for epoch in range(start_epoch, args.epochs): 
                    clf_loss, acc, acc_l, hs = fix(args, train_loader, model, clf, (optimizer, optimizer_phi), epoch, train=True)
                    print('-' * 90)
                    meta = "| epoch {:2d} ".format(epoch)
                    print(meta + "| Train loss {:5.2f} | Train acc {:5.2f} | L acc {:5.2f} | hs {:5.2f}".format(clf_loss, acc, acc_l, hs))

        except KeyboardInterrupt:
            print('-'*50)
            print('Quit Training')

        test_loss, test_acc, test_acc_l, hs = fix(args, test_loader, model, clf, (optimizer, optimizer_phi), epoch, False)
        print('-'*50)
        print("| Test loss {:5.2f} | Test acc {:5.2f} | L acc {:5.2f} | hs {:5.2f}".format(test_loss, test_acc, test_acc_l, hs))

        optimizer_c2 = torch.optim.Adam(clf2.parameters(), lr=args.lr)
        for epoch in range(args.lepochs):
            clf2.train()
            criterion = torch.nn.CrossEntropyLoss()
            correct = 0 
            total = 0
            hs = 0
            for i, (data, label, light) in enumerate(train_loader):
                data, label, light = data[:, 0, :, :].to(args.device), label.to(args.device), light.to(args.device)
                _, z = model(data.view(-1, 32*32))
                phi = model.phi(z)
                l = clf2(F.relu(z.detach()))
                loss = criterion(l, light)
                hsic = HSIC(phi, light)
                optimizer_c2.zero_grad()
                loss.backward()
                optimizer_c2.step()
                hs += hsic.item()
                total += data.size(0)
                _, predicted = torch.max(l.data, 1)
                correct += (predicted == light).sum().item()
            acc = 100 * correct / total
            print('-'*50)
            print("adv {:5.2f} | hs {:5.2f}".format(acc, hs))
        clf2.eval()
        correct = 0
        total = 0
        hs = 0
        for i, (data, label, light) in enumerate(test_loader):
            data, label, light = data[:, 0, :, :].to(args.device), label.to(args.device), light.to(args.device)
            _, z = model(data.view(-1, 32*32))
            l = clf2(F.relu(z.detach()))
            phi = model.phi(z)
            hsic = HSIC(phi, light)
            hs += hsic.item()
            total += data.size(0)
            _, predicted = torch.max(l.data, 1)
            correct += (predicted == light).sum().item()
        acc = 100 * correct / total
        print('-'*50)
        print("adv {:5.2f} | hs {:5.2f}".format(acc, hs))
    else:
        try:
            if args.pretrain:
                criterion = torch.nn.CrossEntropyLoss()
                for epoch in range(30):
                    for i, (data, label, light) in enumerate(train_loader):
                        data, label, light = data[:, 0, :, :].to(args.device), label.to(args.device), light.to(args.device)
                        y, z = model(data.view(-1, 32*32))
                        optimizer.zero_grad()
                        loss = criterion(y, label)
                        loss.backward()
                        optimizer.step()
                for epoch in range(30):
                    for i, (data, label, light) in enumerate(train_loader):
                        data, label, light = data[:, 0, :, :].to(args.device), label.to(args.device), light.to(args.device)
                        y, z = model(data.view(-1, 32*32))
                        phi = model.phi(z)
                        optimizer_phi.zero_grad()
                        phi = model.phi(z.detach())
                        neg_h = -HSIC(phi, light)
                        neg_h.backward()
                        optimizer_phi.step()
            for epoch in range(start_epoch, args.epochs): 
                clf_loss, acc, acc_l, hs = run(args, train_loader, model, clf, (optimizer, optimizer_phi), epoch, train=True)
                print('-' * 90)
                meta = "| epoch {:2d} ".format(epoch)
                print(meta + "| Train loss {:5.2f} | Train acc {:5.2f} | L acc {:5.2f} | hs {:5.2f}".format(clf_loss, acc, acc_l, hs))

        except KeyboardInterrupt:
            print('-'*50)
            print('Quit Training')

        test_loss, test_acc, test_acc_l, hs = run(args, test_loader, model, clf, (optimizer, optimizer_phi), epoch, False)
        print('-'*50)
        print("| Test loss {:5.2f} | Test acc {:5.2f} | L acc {:5.2f} | hs {:5.2f}".format(test_loss, test_acc, test_acc_l, hs))

        optimizer_c2 = torch.optim.Adam(clf2.parameters(), lr=args.lr)
        for epoch in range(args.lepochs):
            clf2.train()
            criterion = torch.nn.CrossEntropyLoss()
            correct = 0 
            total = 0
            hs = 0
            for i, (data, label, light) in enumerate(train_loader):
                data, label, light = data[:, 0, :, :].to(args.device), label.to(args.device), light.to(args.device)
                _, z = model(data.view(-1, 32*32))
                phi = model.phi(z)
                l = clf2(F.relu(z.detach()))
                loss = criterion(l, light)
                hsic = HSIC(phi, light)
                optimizer_c2.zero_grad()
                loss.backward()
                optimizer_c2.step()
                hs += hsic.item()
                total += data.size(0)
                _, predicted = torch.max(l.data, 1)
                correct += (predicted == light).sum().item()
            acc = 100 * correct / total
            print('-'*50)
            print("adv {:5.2f} | hs {:5.2f}".format(acc, hs))
        clf2.eval()
        correct = 0
        total = 0
        hs = 0
        for i, (data, label, light) in enumerate(test_loader):
            data, label, light = data[:, 0, :, :].to(args.device), label.to(args.device), light.to(args.device)
            _, z = model(data.view(-1, 32*32))
            l = clf2(F.relu(z.detach()))
            phi = model.phi(z)
            hsic = HSIC(phi, light)
            hs += hsic.item()
            total += data.size(0)
            _, predicted = torch.max(l.data, 1)
            correct += (predicted == light).sum().item()
        acc = 100 * correct / total
        print('-'*50)
        print("adv {:5.2f} | hs {:5.2f}".format(acc, hs))


if __name__ == "__main__":
    main(args)
    