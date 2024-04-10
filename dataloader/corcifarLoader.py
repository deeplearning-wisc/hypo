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



class CorCIFARDataset(data.Dataset):
    def __init__(self, ood_loc, set_name, cortype, dataset):

        if dataset == 'CIFAR-10':
            print('loading CorCIFAR-10')

            # CorCIFAR_train_path = './data/cifar10_trainc'
            # CorCIFAR_test_path = './data/cifar10_testc'
            CorCIFAR_train_path = f'{ood_loc}/CorCIFAR10_train'
            CorCIFAR_test_path = f'{ood_loc}/CorCIFAR10_test'
            self.num_class = 10

        # elif dataset in ['CIFAR-100']:
        #     print('loading CorCIFAR-100')

        #     CorCIFAR_train_path = './data/cifar100_trainc'
        #     CorCIFAR_test_path = './data/cifar100_testc'  
        #     self.num_class = 100

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        self._image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        if set_name == 'train':
            images=np.load(os.path.join(CorCIFAR_train_path, cortype + '.npy'))
            labels = np.load(os.path.join(CorCIFAR_train_path, 'labels.npy'))
        elif set_name == 'test':
            images = np.load(os.path.join(CorCIFAR_test_path, cortype + '.npy'))
            labels = np.load(os.path.join(CorCIFAR_test_path, 'labels.npy'))

        self.data = images
        self.label = labels

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = self._image_transformer(img)
        
        return img, label

    def __len__(self):
        return len(self.data)










