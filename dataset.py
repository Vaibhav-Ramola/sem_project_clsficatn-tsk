import string
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class GSQDataset(Dataset):
    def __init__(self, img_path, labels_path, transforms = None) -> None:
        super().__init__()
        self.img_apth = img_path
        self.labels_path = labels_path
        self.transforms = transforms
        self.images = np.load(img_path)
        self.labels = np.load(labels_path)

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transforms:
            image = self.transforms(image)
            
        return (image, label)
