import numpy as np
from sklearn import datasets
import torch
from torch.utils.data import Dataset, DataLoader
from traitlets.traitlets import Bool

class Sklearn_Dataset(Dataset):
    def __init__(self, name:str, labels:tuple=(0,1), device:torch.device=torch.device('cpu')) -> None:
        """
        Args:
            name (str): type of the dataset (i.e. 'iris', 'wine')
            num (int): size of dataset
            device (torch.device): device to save target and data tensors
        """
        self.labels = labels
        self.ds_name = name
        if name == 'iris':
            ds = datasets.load_iris()
        elif name == 'wine':
            ds = datasets.load_wine()
        else:
            raise NoSuchDataset
        data = ds.data
        target = ds.target
        _temp =[t in labels for t in target]
        self.data = torch.tensor(data[_temp, :], dtype=torch.float, requires_grad=False, device=device)
        self.target = torch.tensor(target[_temp], dtype=torch.float, requires_grad=False, device=device)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def get_sets(self, num_data:int, shuffle:Bool=True):
        dl = DataLoader(self, batch_size=num_data, shuffle=shuffle)
        Xy = tuple(dl)[0]
        X = Xy[0]
        y = torch.empty_like(Xy[1])
        y[Xy[1]<np.mean(self.labels)]=-1
        y[Xy[1]>=np.mean(self.labels)]=1
        Xtyt = tuple(dl)[1]
        Xt = Xtyt[0]
        yt = torch.empty_like(Xtyt[1])
        yt[Xtyt[1]<np.mean(self.labels)]=-1
        yt[Xtyt[1]>=np.mean(self.labels)]=1

        if self.data.get_device() < 0:# if device is cpu,
            X = np.array( X.numpy(), dtype=float)
            y = np.array( y.numpy(), dtype=float)
            Xt =  np.array(Xt.numpy(), dtype=float)
            yt =  np.array(yt.numpy(), dtype=float)
        
        return X, y, Xt, yt

class NoSuchDataset:
    pass