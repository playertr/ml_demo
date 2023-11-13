import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from omegaconf import DictConfig

def get_dataloader(cfg: DictConfig) -> DataLoader:

    # Load MNIST dataset.
    train_ds = MNIST(
        os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
    )

    # Only take a subset of the data for faster training.
    indices = torch.arange(32)
    train_ds = Subset(train_ds, indices)
    train_loader = DataLoader(train_ds, batch_size=8)

    return train_loader
