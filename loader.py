import numpy as np
import torch
from torch.utils.data.dataset import Dataset



class MicroscopyData(Dataset):
    def __init__(self, directory_path, start, end, which = -1):
        self.which = which
        self.start = start
        self.end = end
        self.path = directory_path
            

    def __getitem__(self,index):
        adjusted_index = index + self.start
        if(self.which == -1):
            label = torch.from_numpy(np.load(self.path + "/13/" + str(index) + ".npy")).float()
            features = []
            for i in range(13):
                features.append(torch.from_numpy(np.load(self.path + "/" +str(i) + "/" + str(index) + ".npy")).float())
            return features, label
	elif(self.which == -2):
	    label = torch.from_numpy(np.load(self.path + "/13/" + str(index) + ".npy")).float()
            features = []
            for i in range(13):
                features.append(np.load(self.path + "/" +str(i) + "/" + str(index) + ".npy"))
	    features = torch.from_numpy(np.array(features)).float()
            return features, label
        return torch.from_numpy(np.load(self.path + "/" +str(self.which) + "/" + str(index) + ".npy")).float()
    def __len__(self):
        return self.end - self.start +1

"""
dataset = MicroscopyData('data',0,14,0)
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=15)
from utils import *
x1, _ = next(iter(loader))
grid = arr_to_2d( np.array(x1), 5, 16)
result = generate_grid(grid)
from PIL import Image
x = Image.fromarray(result*255).convert("RGB")
x.show()
"""
