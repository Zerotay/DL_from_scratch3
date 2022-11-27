import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from sklearn.metrics import accuracy_score


class MLP(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        preds = self.layers(x)
        return preds
    

model = MLP(784).to('cuda')


train_dataset = MNIST(
    root='dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

train_dataset, valid_dataset = \
    random_split(
        train_dataset, [0.8, 0.2], 
        generator=torch.Generator().manual_seed(42)
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=8,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=8,
)

test_dataset = MNIST(
    root='dataset/',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)


train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=8
)


test_loader = DataLoader(
    test_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=8
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    accs = []
    model.train()
    for x, y in train_loader:
        x = x.view(x.size(0), -1).to('cuda')
        y = y.to('cuda')
        y_hat = model(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('train_loss:', loss.item())

    model.eval()
    for x, y in valid_loader:
        x = x.view(x.size(0), -1).to('cuda')
        y = y.to('cuda')
        y_hat = model(x)
        _, preds = y_hat.max(1)

        acc = accuracy_score(preds.detach().cpu().tolist(), y.detach().cpu().tolist())
        accs.append(acc)
    
    print('valid_acc:', np.mean(accs))