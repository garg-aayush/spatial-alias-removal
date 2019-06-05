import os 
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.io import loadmat


class Data(Dataset):
    def __init__(self, filename_x='data_25', filename_y='data_125', directory = "Data/", transforms = transforms.ToTensor()):
        #loading data
        x = loadmat(os.path.join('Data/'+filename_x))[filename_x]
        y = loadmat(os.path.join('Data/'+filename_y))[filename_y]
        #transform makes sure that type is torch and that the dimensions are (NxHxW)
        x_transformed = transforms(x)
        y_transformed = transforms(y)
        self.data = {'X':x_transformed.unsqueeze_(1), 'Y':y_transformed.unsqueeze_(1)}
        
    def __len__(self):
        return self.data['X'].shape[0]
        
    def __getitem__(self, idx):
        return {'x': self.data['X'][idx],'y': self.data['Y'][idx] }