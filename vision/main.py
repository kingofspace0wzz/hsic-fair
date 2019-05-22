import argparse
import time
import random
import math
import torch
import torch.nn.functional as F

from model import VAE, Discriminator
from data import get_mnist, get_cifar, get_fashion_mnist, get_stl10, get_svhn, get_yale
from utils import batch_rotate
from loss import (recon_loss, total_kld, disc_loss)


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/mnist',
                    help='location of the data')
parser.add_argument('--batch_size', type=int, default=128),
parser.add_argument('--epochs', type=int, default=100),
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
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

rotations = ('0', '90', '180', '270')

def baseline(args, data_iter, model, optimizer, epoch, train=True):
    batch_size = args.batch_size
    size = len(data_iter.dataset)
    if train:
        model.train()
    else:
        model.eval()
    # data_iter.init_epoch()
    re_loss = 0
    r_re_loss = 0
    kl_divergence = 0
    r_kl_divergence = 0
    discriminator_loss = 0
    nll = 0
    for i, (data, label) in enumerate(data_iter):
        data = data.to(args.device)
        disloss = torch.zeros(1).to(args.device)

        if train:
            recon, q_z, p_z, z = model(data)
            recon = recon.view(-1, data.size(-2), data.size(-1))
            reloss = recon_loss(recon, data) # sum over batch
            kld = total_kld(q_z, p_z) # sum over batch
            optimizer.zero_grad()
            loss = (reloss + kld) / batch_size
            loss.backward()
            optimizer.step()
        else:
            angles = torch.randint(0, 3, (data.size(0), )).to(args.device)
            r_data = batch_rotate(data.clone(), angles)
            r_recon, r_qz, r_pz, r_z = model(r_data)
            r_recon = r_recon.view(-1, 1, data.size(-2), data.size(-1))
            reloss = recon_loss(r_recon, r_data)
            kld = total_kld(r_qz, r_pz)

        re_loss += reloss.item() / size
        kl_divergence += kld.item() / size
        discriminator_loss += disloss.item() / size
    
    nll = re_loss + kl_divergence
    return nll, re_loss, kl_divergence, discriminator_loss

def run(args, data_iter, model, D, optimizer, optimizer_D, epoch, train=True):
    batch_size = args.batch_size
    size = len(data_iter.dataset)
    if train:
        model.train()
    else:
        model.eval()
    # data_iter.init_epoch()
    re_loss = 0
    r_re_loss = 0
    kl_divergence = 0
    r_kl_divergence = 0
    discriminator_loss = 0
    nll = 0
    total = 0
    correct = 0
    for i, (data, label) in enumerate(data_iter):
        data = data.to(args.device)
        recon, q_z, p_z, z = model(data)
        recon = recon.view(-1, data.size(-2), data.size(-1))
        reloss = recon_loss(recon, data) # sum over batch
        kld = total_kld(q_z, p_z) # sum over batch
        disloss = torch.zeros(1).to(args.device)

        if train:
            if args.ro:
                r_reloss, r_kld = [], []
                for d in range(1, len(rotations)):
                    angles = torch.tensor([d], dtype=torch.long, device=args.device).expand(data.size(0))
                    r_data = batch_rotate(data.clone(), angles)
                    r_recon, r_qz, r_pz, r_z = model(r_data)
                    r_recon = r_recon.view(-1, 1, data.size(-2), data.size(-1))
                    if args.D:
                        D_z = D(r_z)
                        disloss += disc_loss(D_z, angles) # sum over batch
                    r_reloss.append(recon_loss(r_recon, r_data))
                    r_kld.append(total_kld(r_qz, r_pz))
            
                reloss += sum(r_reloss) # / (len(rotations)-1)
                kld += sum(r_kld) # / (len(rotations)-1)
                
                if args.D:
                    optimizer_D.zero_grad()
                    D_loss = disloss / batch_size
                    D_loss.backward(retain_graph=True)
                    optimizer_D.step()

                optimizer.zero_grad()
                loss = (reloss + kld - disloss) / batch_size
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                loss = (reloss + kld) / batch_size
                loss.backward()
                optimizer.step()
        else:
            angles = torch.randint(0, 3, (data.size(0), )).to(args.device)
            r_data = batch_rotate(data.clone(), angles)
            r_recon, r_qz, r_pz, r_z = model(r_data)
            r_recon = r_recon.view(-1, 1, data.size(-2), data.size(-1))
            reloss = recon_loss(r_recon, r_data)
            kld = total_kld(r_qz, r_pz)

            # accuracy of the discriminator
            output = D(r_z)
            _, predicted = torch.max(output.data, 1)
            total += angles.size(0)
            correct += (predicted == angles).sum().item()

        re_loss += reloss.item() / size
        kl_divergence += kld.item() / size
        discriminator_loss += disloss.item() / size
        # accuracy = 100 * correct / total
    
    nll = re_loss + kl_divergence
    return nll, re_loss, kl_divergence, discriminator_loss

def main(args):
    print("Loading data")
    dataset = args.data.rstrip('/').split('/')[-1]
    if dataset in ['mnist']:
        train_loader, test_loader = get_mnist(args.batch_size, args.data)
    elif dataset in ['cifar']:
        train_loader, test_loader, classes = get_cifar(args.batch_size, args.data)
    elif dataset in ['svhn']:
        train_loader, test_loader, extra_loader = get_svhn(args.batch_size, args.data)
    elif dataset in ['fashion']:
        train_loader, test_loader = get_fashion_mnist(args.batch_size, args.data)
    elif dataset in ['stl10']:
        train_loader, test_loader, unlabeled_loader = get_stl10(args.batch_size, args.data)
    elif dataset in ['yale']:
        train_loader = get_yale(args.batch_size, args.data)
    else:
        raise NotImplementedError
    torch.cuda.set_device(args.device_id)
    for _, (batch, _) in enumerate(train_loader):
        size = batch.size()
        break

    model = VAE(size, args.code_dim, args.batch_size, data=dataset).to(args.device)
    D = Discriminator(args.code_dim, 4).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer_D = torch.optim.Adam(model.parameters(), args.lr)

    start_epoch = 1
    print('\nStarting Training')
    if args.base:
        try:
            for epoch in range(start_epoch, args.epochs): 
                nll, re_loss, kl_divergence, d_loss = baseline(args, train_loader, model, optimizer, epoch, train=True)
                print('-' * 90)
                meta = "| epoch {:2d} ".format(epoch)
                print(meta + "| Train NLL: {:5.2f} | Train loss: {:5.2f} ({:5.2f}) | D loss {:5.2f} |".format(nll, re_loss, kl_divergence, d_loss))

                nll, re_loss, kl_divergence, d_loss = baseline(args, test_loader, model, optimizer, 1, train=False)
                print(len(meta)* " " + "| Test NLL: {:5.2f} | Test loss: {:5.2f} ({:5.2f}) | D loss {:5.2f} |".format(nll, re_loss, kl_divergence, d_loss))

        except KeyboardInterrupt:
            print('-'*50)
            print('Quit Training')

        nll, re_loss, kl_divergence, d_loss = baseline(args, test_loader, model, optimizer, epoch, train=False)    
        print('='* 90)
        print("| Train NLL: {:5.2f} | Train loss: {:5.2f} ({:5.2f}) | D loss {:5.2f} |".format(nll, re_loss, kl_divergence, d_loss))

    else:
        try:
            for epoch in range(start_epoch, args.epochs): 
                nll, re_loss, kl_divergence, d_loss = run(args, train_loader, model, D, optimizer, optimizer_D, epoch, train=True)
                print('-' * 90)
                meta = "| epoch {:2d} ".format(epoch)
                print(meta + "| Train NLL: {:5.2f} | Train loss: {:5.2f} ({:5.2f}) | D loss {:5.2f} |".format(nll, re_loss, kl_divergence, d_loss))

                nll, re_loss, kl_divergence, d_loss = run(args, test_loader, model, D, optimizer, optimizer_D, 1, train=False)
                print(len(meta)* " " + "| Test NLL: {:5.2f} | Test loss: {:5.2f} ({:5.2f}) | D loss {:5.2f} |".format(nll, re_loss, kl_divergence, d_loss))

        except KeyboardInterrupt:
            print('-'*50)
            print('Quit Training')

        nll, re_loss, kl_divergence, d_loss = run(args, test_loader, model, D, optimizer, optimizer_D, epoch, train=False)    
        print('='* 90)
        print("| Train NLL: {:5.2f} | Train loss: {:5.2f} ({:5.2f}) | D loss {:5.2f} |".format(nll, re_loss, kl_divergence, d_loss))

if __name__ == "__main__":
    main(args)