import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Some Information about VAE"""
    def __init__(self):
        super(VAE, self).__init__()

    def forward(self, x):

        return x

class SVM(nn.Module):
    """Some Information about SVM"""
    def __init__(self, input_dim, code_dim, hidden, batch_size, num_labels, num_factors, dropout=0.2):
        super(SVM, self).__init__()
        self.encoder = Encoder(input_dim, hidden, code_dim, dropout)
        self.alpha = nn.Parameter(torch.zeros(batch_size))

    def forward(self, x, batch):
        z = self.encoder(x)
        z_b = self.encoder(x)

        return out

class Encoder(nn.Module):
    """Some Information about Encoder"""
    def __init__(self, n_features, hidden, code_dim, dropout=0.2):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden, hidden),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden, code_dim),
            # nn.Linear(n_features, code_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Classifier(nn.Module):
    """Some Information about Classifier"""
    def __init__(self, input_dim, num_labels, hidden_dim=64):
        super(Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.map = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim, num_labels),
        )
       
    def forward(self, x):
        phi = self.map(x)
        output = self.net(x)
        return output, phi

class Model(nn.Module):
    """Some Information about Model"""
    def __init__(self, input_dim, code_dim, hidden, num_labels, num_factors, dropout=0.2):
        super(Model, self).__init__()
        self.encoder = Encoder(input_dim, hidden, code_dim, dropout)
        # self.classifier = nn.Linear(code_dim, num_labels)
        self.classifier = Classifier(code_dim, num_labels, hidden)
        self.gender = nn.Linear(code_dim, num_labels)
        self.race = nn.Linear(code_dim, num_labels)

    def forward(self, x):
        z = self.encoder(x)
        y, phi = self.classifier(F.relu(z))
        # r = self.race(z.detach()) 
        # g = self.gender(z.detach())
        return y, z, phi
