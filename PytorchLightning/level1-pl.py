import os

import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from torchmetrics import Accuracy


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
    

class MLPLightning(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
    
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        preds = y_hat.max(1)[1].long()
        self.train_acc(preds, y)
        return loss
    

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        preds = y_hat.max(1)[1].long()
        self.valid_acc(preds, y)
        self.log_dict(
            {
                'valid_loss': loss,
                'valid_acc': self.valid_acc
            },
            on_epoch=True,
            prog_bar=True
        )
        return loss
    

    def on_validation_epoch_end(self) -> None: # 없어도
        self.log('valid_acc', self.valid_acc, on_epoch=True, prog_bar=True)


    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.model(x)
        return y_hat.max(1)[1].long()
    

    def on_predict_epoch_end(self, results) -> None:
        write_path = os.path.join(
            '.', 
            f"{self.model.__class__.__name__}_submission.csv"
        )

        total_preds = torch.cat(results[0])
        

        with open(write_path, "w", encoding="utf8") as w:
            w.write("id,prediction\n")
            for id, p in enumerate(total_preds):
                w.write("{},{}\n".format(id, p))
        
        return total_preds

    
    
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

torch_model = MLP(784)
ligthning_model = MLPLightning(torch_model)

trainer = pl.Trainer(
    max_epochs=1,
    log_every_n_steps=10
)

trainer.fit(
    model=ligthning_model,
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader,
    
)

trainer.predict(
    model=ligthning_model,
    dataloaders=test_loader
)
