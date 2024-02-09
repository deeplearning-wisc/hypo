import numpy as np
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from random import sample, random


from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset


import os.path as osp
from torch.utils.data import Dataset
from tqdm import tqdm

import os

from random import sample, random

ROOT_PATH = '/datasets/VLCSDataset/'

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VLCSDataset(data.Dataset):
    def __init__(self, setname, target_domain, augment_full = True):

        self._image_transformer_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if augment_full:
            self._image_transformer_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else: 
            self._image_transformer_train = self._image_transformer_test

        if setname != 'test':
            fulldata = []
            label_name = []
            fullconcept = []
            i = 0
            domain = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
            domain.remove(target_domain)
            for domain_name in domain:
                txt_path = os.path.join(ROOT_PATH, domain_name + '.txt')
                images, labels = self._dataset_info(txt_path)
                concept = [i] *len(labels)
                fulldata.extend(images)
                label_name.extend(labels)
                fullconcept.extend(concept)
                i += 1

            name_train, name_val, labels_train, labels_val, concept_train, concept_val = self.get_random_subset(fulldata, label_name, fullconcept, 0.1)

            if setname == 'train':
                self.data = name_train
                self.label = labels_train
                self.concept = concept_train
            elif setname == 'val':
                self.data = name_val
                self.label = labels_val
                self.concept = concept_val
        else:
            domain_name = target_domain
            txt_path = os.path.join(ROOT_PATH, domain_name + '.txt')
            self.data, self.label = self._dataset_info(txt_path)
            self.concept = [-1]*len(self.label)

        self.setname = setname


    def _dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            path = os.path.join(row[0])
            path = path.replace('\\', '/')

            file_names.append(path)
            labels.append(int(row[1]))

        return file_names, labels

    def get_random_subset(self, names, labels, concepts, percent):
        """
        :param names: list of names
        :param labels:  list of labels
        :param percent: 0 < float < 1
        :return:
        """
        samples = len(names)
        amount = int(samples * percent)
        random_index = sample(range(samples), amount)
        name_val = [names[k] for k in random_index]
        name_train = [v for k, v in enumerate(names) if k not in random_index]
        labels_val = [labels[k] for k in random_index]
        labels_train = [v for k, v in enumerate(labels) if k not in random_index]
        concepts_val = [concepts[k] for k in random_index]
        concepts_train = [v for k, v in enumerate(concepts) if k not in random_index]

        return name_train, name_val, labels_train, labels_val, concepts_train, concepts_val




    def __getitem__(self, index):
        data, label, concept= self.data[index], self.label[index], self.concept[index]

        _img = Image.open(data).convert('RGB')
        if self.setname == 'val' or self.setname == 'test':
            img = self._image_transformer_test(_img)
            return img, label, concept
        else:
            img = self._image_transformer_train(_img)
            return img, label, concept

    def __len__(self):
        return len(self.data)

