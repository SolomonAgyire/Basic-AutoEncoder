from pytorch_lightning import Trainer, loggers
from model import Autoencoder
from dataloader import get_dataloader
import os

def train_model():
    file_path = r"C:\Users\sagyi\Downloads\x1.txt"
    
    train_loader, val_loader = get_dataloader(file_path, batch_size=32)

    model = Autoencoder()

    # TensorBoard logger
    tb_logger = loggers.TensorBoardLogger('runs/')

    trainer = Trainer(
        max_epochs=50,
        logger=tb_logger,
        log_every_n_steps=5,
        check_val_every_n_epoch=1
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    train_model()
