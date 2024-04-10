import numpy as np
import os
import pickle
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision

import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pathlib

'''
This script makes the datasets used in eval cifar. The main function is make_datasets. 
'''



def load_CIFAR(id_loc, dataset, classes=[]):

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
    #                                trn.ToTensor(), trn.Normalize(mean, std)])
    train_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    if dataset == 'CIFAR-10':
        cifar10_path = f'{id_loc}/cifar10'
        print('loading CIFAR-10')
        train_data = dset.CIFAR10(
            cifar10_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(
            cifar10_path, train=False, transform=test_transform, download=True)

    # elif dataset in ['CIFAR-100']:
    #     cifar100_path = f'{id_loc}/cifar100'
    #     print('loading CIFAR-100')
    #     train_data = dset.CIFAR100(
    #         cifar100_path, train=True, transform=train_transform, download=True)
    #     test_data = dset.CIFAR100(
    #         cifar100_path, train=False, transform=test_transform, download=True)

    return train_data, test_data

def load_CorCifar(ood_loc, dataset, cortype):

    if dataset == 'CIFAR-10':
        print('loading CorCIFAR-10')

        from dataloader.corcifarLoader import CorCIFARDataset as Dataset

        train_data = Dataset(ood_loc, 'train', cortype, dataset)
        test_data = Dataset(ood_loc, 'test', cortype, dataset)

    # elif dataset in ['CIFAR-100']:
    #     print('loading CorCIFAR-100')

    #     from dataloader.corcifar100Loader import CorCIFARDataset as Dataset

    #     train_data = Dataset('train', cortype, dataset)
    #     test_data = Dataset('test', cortype, dataset)

    return train_data, test_data

def make_datasets(id_loc, ood_loc, in_dset, state, cortype):
    #rng = np.random.default_rng(state['seed'])

    print('building datasets...')
    train_in_data, test_in_data = load_CIFAR(id_loc, in_dset)
    train_cor_data, test_cor_data = load_CorCifar(ood_loc, in_dset, cortype)

    test_loader_in = torch.utils.data.DataLoader(
        test_in_data,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    test_loader_cor = torch.utils.data.DataLoader(
        test_cor_data,
        batch_size=state['batch_size'], shuffle=False,
        num_workers=state['prefetch'], pin_memory=True)

    return test_loader_in, test_loader_cor
