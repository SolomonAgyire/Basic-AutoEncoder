import torch
from torch import nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class Autoencoder(pl.LightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(250, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 250),
            nn.Sigmoid() 
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'inputs': inputs, 'outputs': outputs}

    def on_validation_epoch_end(self):
        val_dataloader = self.trainer.val_dataloaders

        # Check if DataLoader has batches
        #if not any(val_dataloader):
            #print("Validation DataLoader is empty. Skipping visualization.")
           #return

        # first batch
        for sample_batch in val_dataloader:
            sample_inputs, _ = sample_batch  
            with torch.no_grad():
                reconstructed = self.forward(sample_inputs)

            # Visualize reconstruction
            fig = self.visualize_reconstruction(sample_inputs, reconstructed)
            if self.logger is not None:
                self.logger.experiment.add_figure('Reconstructed pulses', fig, global_step=self.current_epoch)
            plt.close(fig)

            # Only first batch for visualization
            break

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    #Reconstruction
    def visualize_reconstruction(self, inputs, outputs):
        inputs = inputs.detach().cpu()
        outputs = outputs.detach().cpu()
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
        for i, ax in enumerate(axes.flatten()):
            if i >= inputs.size(0):
                break
            ax.plot(inputs[i], 'b-', label='Original')
            ax.plot(outputs[i], 'r--', label='Reconstructed')
            ax.set_title(f'Pulse {i+1}')
            ax.legend()
        return fig
