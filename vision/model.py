import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Encoder(nn.Module):
    def __init__(self, code_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, code_dim*2)
        )
        self.code_dim = code_dim

    def forward(self, inputs):
        p_z = Normal(torch.zeros((inputs.size(0), self.code_dim), device=inputs.device),
                      (0.5 * torch.zeros((inputs.size(0), self.code_dim), device=inputs.device)).exp())
        if inputs.size(-1) != 784:
            inputs = inputs.view(-1, 784)
        h = self.encoder(inputs)
        mu = h[:, self.code_dim:]
        logvar = h[:, :self.code_dim]
        return Normal(mu, (0.5 * logvar).exp()), p_z

class Decoder(nn.Module):
    def __init__(self, code_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 784)
        )

    def forward(self, z):
        h = self.decoder(z)
        return torch.sigmoid(h)

class ConvEncoder(nn.Module):
    def __init__(self, code_dim, nc=1):
        super().__init__()
        self.nc = nc
        self.code_dim = code_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, kernel_size=4, stride=1),
            nn.ReLU(True),
            View((-1, 256)),
            nn.Linear(256, code_dim * 2)
        )

    def forward(self, inputs):
        p_z = Normal(torch.zeros((inputs.size(0), self.code_dim), device=inputs.device),
                      (0.5 * torch.zeros((inputs.size(0), self.code_dim), device=inputs.device)).exp())
        h = self.encoder(inputs)
        mu = h[:, self.code_dim:]
        logvar = h[:, :self.code_dim]
        return Normal(mu, (0.5 * logvar).exp()), p_z

class ConvDecoder(nn.Module):
    def __init__(self, code_dim, nc=1):
        super().__init__()
        self.nc = nc
        self.code_dim = code_dim
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 256),
            View((-1, 256, 1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 64, kernel_size=4),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, nc, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z):
        outputs = self.decoder(z)
        return outputs

class VAE(nn.Module):
    """VAE"""
    def __init__(self, input_dim, code_dim, batch_size, data='mnist'):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.batch_size = batch_size
        if data == 'mnist':
            self.encoder = Encoder(code_dim)
            self.decoder = Decoder(code_dim)
        elif data == 'cifar':
            self.encoder = ConvEncoder(code_dim, nc=3)
            self.decoder = ConvDecoder(code_dim, nc=3)
        else:
            raise NotImplementedError

    def forward(self, x):
        q_z, p_z = self.encoder(x)
        if self.training:
            z = q_z.rsample()
        else:
            z = q_z.mean
        output = self.decoder(z)
        return output, q_z, p_z, z

class Discriminator(nn.Module):
    '''
    Discriminator that distinguishes different conditional densities q(z|x)
    '''
    def __init__(self, z_dim, num_labels, hidden_dim=1000):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.D = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(0.2, True),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x):
        output = self.D(x)
        return output

class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=1000):
        super(Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, num_labels),
        )
       
    def forward(self, x):
        output = self.net(x)
        return output

class SelfLearner(nn.Module):
    """SelfLearner"""
    def __init__(self, input_dim, num_labels, hidden_dim=1000):
        super(SelfLearner, self).__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, num_labels),
        )
       
    def forward(self, x):
        output = self.net(x)
        return output

class YaleB(nn.Module):
    """Some Information about YaleB"""
    def __init__(self):
        super(YaleB, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(32*32, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
        )
        self.classifier = nn.Linear(128, 38)

    def forward(self, x):
        if x.size(-1) != 32*32:
            x.view(-1, 32*32)
        out = self.classifier(self.net(x))
        return out

class YaleBHSIC(nn.Module):
    """Some Information about YaleBHSIC"""
    def __init__(self):
        super(YaleBHSIC, self).__init__()
        self.encoder = nn.Linear(32*32, 256)
        self.phi = nn.Linear(256, 128)
        self.classifier = nn.Sequential(
            nn.Linear(258, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 38),
        )
    def map(self, z):
        return F.relu(self.phi(F.leaky_relu(z, 0.2, True)))

    def forward(self, x):
        z = self.encoder(x)
        # phi = F.leaky_relu(self.phi(F.leaky_relu(z, 0.5, True)), 0.5, True)
        # phi = self.phi(F.leaky_relu(z, 0.2, True))
        # out = self.classifier(F.leaky_relu(phi))
        
        out = self.classifier(F.leaky_relu(z, 0.2, True))
        return out, z