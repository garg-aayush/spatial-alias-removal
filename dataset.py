"""
Define a Dataset model that inherits from Pytorch's base class.
"""
import os 
import random
import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import Dataset
from scipy.io import loadmat


class Data(Dataset):
    def __init__(self, filename_x='data_25', filename_y='data_125',
                 directory="Data/", transform=transforms.ToTensor()):
        # Loading data.
        x = loadmat(os.path.join(directory, filename_x))[filename_x]
        y = loadmat(os.path.join(directory, filename_y))[filename_y]

        '''
        # Transform makes sure that type is torch and that the
        # dimensions are (NxHxW).
        x_transformed = transforms(x)
        y_transformed = transforms(y)
        '''

        self.transform = transform

        x = x.transpose(2, 0, 1)
        y = y.transpose(2, 0, 1)

        self.data = { 
            'X': x,
            'Y': y
        }

        # Save data shapes for creating models.
        self.input_dim = x.shape[-2:]
        self.output_dim = y.shape[-2:]
        self.output_dim_fk = list(self.output_dim)
        self.output_dim_fk[-1] = self.output_dim_fk[-1] // 2 + 1
        
    def __len__(self):
        return self.data['X'].shape[0]
        
    def __getitem__(self, idx):
        sample = {
            'x': self.data['X'][idx],
            'y': self.data['Y'][idx]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        return {
            'x': torch.from_numpy(x.copy()).unsqueeze(0),
            'y': torch.from_numpy(y.copy()).unsqueeze(0)
        }


class RandomHorizontalFlip(object):
    def __init__(self, flip_p=0.5):
        """
        Randomly flip a sample horizontally.
        """
        self.flip_p = flip_p

    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        if random.random() < self.flip_p:
            x = np.fliplr(x)
            y = np.fliplr(y)

        return {
            'x': x,
            'y': y
        }
