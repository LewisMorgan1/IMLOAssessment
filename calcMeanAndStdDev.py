import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse

# see https://pytorch.org/vision/main/generated/torchvision.datasets.Flowers102.html

# Hyperparameters
NUM_OUT_CH = [8, 16]
IMAGE_W = 200
IMAGE_H = 200
BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 0.001
NUM_WORKERS = 0


# Define the transform without normalization
transform = transforms.Compose([
    transforms.Resize((IMAGE_W, IMAGE_H)),
    transforms.ToTensor()
])

# Load the Flowers102 dataset without normalization
pretrain = torchvision.datasets.Flowers102(root='./flowerdata3', split="train", download=True, transform=transform)

# Initialize variables to store sum and squared sum of pixel values
sum_channel = torch.zeros(3)
sum_channel_squared = torch.zeros(3)

# Iterate over the dataset to compute the sum and squared sum
for images, _ in DataLoader(pretrain, batch_size=64):
    sum_channel += torch.sum(images, dim=[0, 2, 3])
    sum_channel_squared += torch.sum(images ** 2, dim=[0, 2, 3])

# Calculate the mean and standard deviation
num_pixels = len(pretrain) * 224 * 224
mean = sum_channel / num_pixels
std = torch.sqrt((sum_channel_squared / num_pixels) - mean ** 2)

print("Mean:", mean)
print("Standard Deviation:", std)
