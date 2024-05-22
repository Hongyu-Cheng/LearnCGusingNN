import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.IP import *

class KnapsackDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        loaded_dict = torch.load(self.file_paths[index])
        return loaded_dict['ip_tensor'], loaded_dict['treesize']

class KnapsackDatasetMu(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        loaded_dict = np.load(self.file_paths[index],allow_pickle=True).item()
        ip_tensor = torch.from_numpy(loaded_dict['ip_tensor']).float()
        treesize = torch.tensor(loaded_dict['original_treesize']).float()
        return ip_tensor, treesize

class IPDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        loaded_dict = torch.load(self.file_paths[index])
        return loaded_dict['ip_tensor'], loaded_dict['x_LP']