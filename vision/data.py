import numpy as np
import random
import glob
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
# from skimage.io import imread
from torchvision.utils import save_image
import argparse
import scipy.io as sio

def get_mnist(batch_size, path_to_data='data/mnist'):
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])
    train_data = datasets.MNIST(path_to_data, 
                            train=True, download=True, transform=transform)
    test_data = datasets.MNIST(path_to_data, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    return train_loader, test_loader

def get_fashion_mnist(batch_size=128, path_to_data='data/fashion'):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    train_data = datasets.FashionMNIST(path_to_data,
                                       train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(path_to_data,
                                       train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    return train_loader, test_loader

def get_cifar(batch_size, path_to_data='data/cifar'):
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    trainset = datasets.CIFAR10(path_to_data, train=True,
                                                download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(path_to_data, train=False,
                                            download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    return train_loader, test_loader

def get_svhn(batch_size, path_to_data):
    mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    trainset = datasets.SVHN(path_to_data, split='train',
                                                download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

    testset = datasets.SVHN(path_to_data, split='test',
                                            download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    extra = datasets.SVHN(path_to_data, split='extra',
                                            download=True, transform=transform)
    extra_loader = DataLoader(extra, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

    return train_loader, test_loader, extra_loader

def get_stl10(batch_size, path_to_data):
    mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train = datasets.STL10(path_to_data, split='train',
                                                download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    test = datasets.STL10(path_to_data, split='test',
                                                download=True, transform=transform)
    test_loader = DataLoader(test, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    unlabeled = datasets.STL10(path_to_data, split='unlabeled',
                                                download=True, transform=transform)
    unlabeled_loader = DataLoader(unlabeled, batch_size=batch_size,
                                                shuffle=True, num_workers=2) 
    return train_loader, test_loader, unlabeled_loader

def get_yale(batch_size, path):
    all_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    yale = datasets.ImageFolder(path, transform=all_transform)
    train_loader = DataLoader(yale, batch_size=batch_size, shuffle=True)
    return train_loader, train_loader

def get_yale_split(batch_size, path):
    mat_contents = sio.loadmat(path)
    fea = torch.from_numpy(mat_contents['fea']).long()
    gnd = torch.from_numpy(mat_contents['gnd']).long()
    light = torch.ones(64, 1)
    for i in range(64):
        light[i] += int(i/13)
    light = torch.cat([light for i in range(38)], dim=0)
    yale = TensorDataset(fea, gnd, light)
    train_loader = DataLoader(yale, batch_size=args.batch_size, shuffle=True)
    return train_loader
# def get_celeba(batch_size, path_to_data):
#     train = datasets.CelebA

def get_chair(batch_size, path):
    all_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    chair = datasets.ImageFolder(path, transform=all_transform)
    train_loader = DataLoader(chair, batch_size=batch_size, shuffle=True)
    return train_loader, train_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/yale')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    dataset = args.data.rstrip('/').split('/')[-1]
    train_loader = get_yale(args.batch_size, args.data)
    
    # yale = datasets.ImageFolder(args.data, transform=transforms.ToTensor())
    # print(yale.data)

    for i, (data, label) in enumerate(train_loader):
        # print(data[0, 1])
        # print(data[0, 0])
        # print(label)
        print(data.size())
        break
        