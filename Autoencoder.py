import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L


class AutoEncoder(L.LightningModule):
    def __init__(self, input_size, latent_dim, lr):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size,128), 
                                nn.ReLU(),
                                nn.Linear(128,64),
                                nn.ReLU(),
                                nn.Linear(64, latent_dim))
        
        self.decoder = nn.Sequential(nn.Linear(latent_dim,64),
                                nn.ReLU(),
                                nn.Linear(64, 128),
                                nn.ReLU(),
                                nn.Linear(128,input_size),
                                nn.Sigmoid())
        self.lr = lr

    def forward(self,x):
        embedding = self.encoder(x)
        return embedding
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters, lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat,x)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat,x)
        self.log("val_loss", loss)


