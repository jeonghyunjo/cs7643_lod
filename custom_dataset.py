import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Sample lidar dataset class
class LiDARDataset(Dataset):
    def __init__(self, num_samples, num_points):
        self.num_samples = num_samples
        self.num_points = num_points
        self.data = np.random.rand(num_samples, num_points, 3)  # Simulate [N, P, XYZ] dataset
        self.labels = np.random.randint(0, 2, num_samples)  # Binary classification for simplicity

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.long)