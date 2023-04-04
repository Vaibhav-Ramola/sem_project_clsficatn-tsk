import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class GSQDataset(Dataset):
    def __init__(self, img_path, labels_path, transforms = None) -> None:
        super().__init__()
        self.img_apth = img_path
        self.labels_path = labels_path
        self.transforms = transforms
        self.images = np.load(img_path, allow_pickle=True)   
        self.labels = np.load(labels_path, allow_pickle=True)

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if label=='STAR':
            label = np.array([1,0,0])
        elif label=='QSO':
            label = np.array([0,1,0])
        elif label=='GALAXY':
            label = np.array([0,0,1])
        if self.transforms:
            image = self.transforms(image)

        return (image, label)
