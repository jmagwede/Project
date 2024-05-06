import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import pandas as pd
import torchaudio
import boto3
import io
from io import BytesIO



def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    validation_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, validation_data


if __name__ == "__main__":

    # download data and create data loader
    train_data, _ = download_mnist_datasets()
    print("Dataset Downloaded")