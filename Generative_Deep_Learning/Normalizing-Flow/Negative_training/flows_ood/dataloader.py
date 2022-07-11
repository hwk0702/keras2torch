import os
import sys
import random
import argparse
import numpy as np

import torch
from torch import distributions
import torch.nn as nn

import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings('ignore')


class SVHN_(SVHN):
    """
    Reimplementation of SVHN datasets with the same interface as MNIST.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 download=False):
               
        super(SVHN_, self).__init__(root, 
                                    split="train" if train else "test", 
                                    transform=transform, 
                                    target_transform=target_transform,
                                    download=download)
        self.train = train
        if train:
            self.train_data = self.data
            self.train_labels = self.labels
        else:
            self.test_data = self.data
            self.test_labels = self.labels
        delattr(self, "data")
        delattr(self, "labels")
        
    def __getattr__(self, attr):
        if attr == "data":
            if self.train:
                return self.train_data
            else:
                return self.test_data
        elif attr == "labels":
            if self.train:
                return self.train_labels
            else:
                return self.test_labels
        else:
            raise AttributeError(attr)
            
            
def get_transform(args, dataset_name):
    
    if dataset_name in ['cifar10', 'svhn']:
        img_size, n_channels = 32, 3
    elif dataset_name in ['mnist', 'fmnist']:
        img_size, n_channels = 28, 1
    else:
        raise ValueError("Unsupported dataset "+args.dataset)
        
    # define transform
    if args.aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor()
            ])
        
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    
    return transform_train, transform_test, (n_channels, img_size, img_size)


def get_loader(args, dataset, transform_train, transform_test):
    
    if dataset == 'svhn':
        ds = SVHN_
    elif dataset == "fmnist":
        ds = getattr(torchvision.datasets, 'FashionMNIST')
    elif dataset == 'mnist':
        ds = getattr(torchvision.datasets, 'MNIST')
    elif dataset == 'cifar10':
        ds = getattr(torchvision.datasets, 'CIFAR10')
        
    train_set = ds(root=args.data_path, train=True, download=True, transform=transform_train)
    test_set = ds(root=args.data_path, train=False, download=True, transform=transform_test)
    
    if dataset == 'svhn':
        num_classes = max(train_set.train_labels) + 1
    else:
        num_classes = len(train_set.classes)
    
    train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
    test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
    
    return train_loader, test_loader, num_classes


