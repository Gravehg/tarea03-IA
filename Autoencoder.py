import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import matplotlib.pyplot as plt


class AutoEncoder(L.LightningModule):
    def __init__(self, input_size, latent_dim, lr):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size,128), 
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
        self.reconstructed_images = []  # Initialize as empty lists
        self.original_images = []       # Initialize as empty lists

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)

        # Accumulate reconstructed and original images
        self.reconstructed_images.append(x_hat.detach().cpu())
        self.original_images.append(x.detach().cpu())

    def on_validation_epoch_end(self):
        if self.current_epoch % 5 == 0:
            # Concatenate lists of tensors into single tensors for plotting
            reconstructed = torch.cat(self.reconstructed_images, dim=0)
            original = torch.cat(self.original_images, dim=0)
            self.plot_reconstructed_images(original, reconstructed)

        # Clear the lists for the next epoch
        self.reconstructed_images = []
        self.original_images = []

    def plot_reconstructed_images(self, original_images, reconstructed_images):
        num_images = min(original_images.size(0), 10)
        fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
        for i in range(num_images):
            axes[0, i].imshow(original_images[i].cpu().float().numpy().reshape(28, 28), cmap="gray")
            axes[0, i].axis("off")
            axes[1, i].imshow(reconstructed_images[i].cpu().float().numpy().reshape(28, 28), cmap="gray")
            axes[1, i].axis("off")

        plt.show()

    #Test
    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss)
        return loss
