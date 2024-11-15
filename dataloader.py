import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



class PulseDataset(Dataset):
    #File path 
    file_path = r"C:\Users\sagyi\Downloads\x1.txt"
    
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  

def get_dataloader(file_path, batch_size=32, num_workers=4):
    data = pd.read_csv(file_path, sep=' ', header=None).values
    data = data / np.max(data, axis=1, keepdims=True)

    # Split - training and validation 
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # datasets
    train_dataset = PulseDataset(train_data)
    val_dataset = PulseDataset(val_data)

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    return train_loader, val_loader

