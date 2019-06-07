import os 
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.io import loadmat


class Data(Dataset):
    def __init__(self, filename_x='data_25', filename_y='data_125',
                 directory="Data/", transforms=transforms.ToTensor()):
        """
        Dataset object for the seismic data.
        Iterating over it yields dictionaries containing 'x', the low-res
        data measured at intervals of 25 meters and 'y', the high-res data
        measured at the 12.5 meter interval.
        """
        # Loading data.
        x = loadmat(os.path.join(directory, filename_x))[filename_x]
        y = loadmat(os.path.join(directory, filename_y))[filename_y]

        # Transform makes sure that type is torch and that the
        # dimensions are (NxHxW).
        x_transformed = transforms(x)
        y_transformed = transforms(y)

        self.data = {
            'X': x_transformed.unsqueeze_(1).float(),
            'Y': y_transformed.unsqueeze_(1).float()
        }
        
    def __len__(self):
        return self.data['X'].shape[0]
        
    def __getitem__(self, idx):
        return {
            'x': self.data['X'][idx],
            'y': self.data['Y'][idx]
        }
