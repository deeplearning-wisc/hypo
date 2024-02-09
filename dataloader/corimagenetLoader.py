import numpy as np
import torch
import torch.utils.data as data
import torchvision

from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset

import os.path as osp
from torch.utils.data import Dataset
from tqdm import tqdm

import os

from random import sample, random

CorIMAGENET100_train_path = './ImageNet-100-C/train'
CorIMAGENET100_test_path = './ImageNet-100-C/val'

class CorIMAGENETDataset(data.Dataset):
    def __init__(self, set_name, cortype):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self._image_transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.set_name = set_name

        if set_name == 'train':
            images = np.load(os.path.join(CorIMAGENET100_train_path, cortype + '.npy'))
            labels = np.load(os.path.join(CorIMAGENET100_train_path, 'labels.npy'))
        elif set_name == 'test':
            images = np.load(os.path.join(CorIMAGENET100_test_path, cortype + '.npy'))
            labels = np.load(os.path.join(CorIMAGENET100_test_path, 'labels.npy'))

        self.data = images
        self.label = labels

        self.num_class = 100

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = self._image_transformer(img)
       
        return img, label

    def __len__(self):
        return len(self.data)










