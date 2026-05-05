import torch
import numpy as np
import os
import sys
import random
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

from torch.utils.data import sampler
from torch.utils.data.sampler import SubsetRandomSampler
from cub import CubDataset
from flower import FlowerDataset
from torchvision import transforms
from torch.utils.data import DataLoader

def data_class(dataset = 'cifar10'):
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'flower':
        num_classes = 102
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'cub':
        num_classes = 200
    return num_classes

def dataloader(dataset = 'cifar10', data_mode = 'train', batch_size = 32, num_workers = 8, image_size = 224):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose([
            # transforms.Resize((args.img_size, args.img_size)),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    # 设置随机数种子
    torch.manual_seed(42)
    if data_mode == 'train':
        if dataset == 'cifar10':
            valid = datasets.CIFAR10(root=r'F:\QML\Papers\CODES\FQN_HEDL\Data\CIFAR10', train=True, download=True, transform=train_transform)
        elif dataset == 'cifar100':
            valid = datasets.CIFAR100(root='/home/miyu/桌面/shift-wiseConv-master/cifar-100-python', train=True, download=False, transform=train_transform)
        elif dataset == 'flower':
            valid = FlowerDataset(transform=train_transform,data_mode='train')
        elif dataset == 'cub':
            valid = CubDataset(transform=train_transform,data_mode='train')
        elif dataset == 'SVHN':
            valid = datasets.SVHN(root='F:\QML\Papers\CODES\hedl\datasets\SVHN', split='train', download=True, transform=train_transform)
        elif dataset == 'place365':
            valid = datasets.Places365(root='/mnt/Data/place365', split = 'train-standard', download=False, transform=train_transform)
        else:
            print('Wrong dataset name!')
            sys.exit(0)
        
        valid_dl = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    elif data_mode == 'valid':
        if dataset == 'cifar10':
            valid = datasets.CIFAR10(root=r'F:\QML\Papers\CODES\FQN_HEDL\Data\CIFAR10', train=False, download=True, transform=test_transform)
        elif dataset == 'cifar100':
            valid = datasets.CIFAR100(root='/home/miyu/桌面/shift-wiseConv-master/cifar-100-python', train=False, download=False, transform=test_transform)
        elif dataset == 'flower':
            valid = FlowerDataset(transform=test_transform,data_mode='valid')
        elif dataset == 'cub':
            valid = CubDataset(transform=test_transform,data_mode='valid')
        elif dataset == 'SVHN':
            valid = datasets.SVHN(root='F:\QML\Papers\CODES\hedl\datasets\SVHN', split='test', download=True, transform=test_transform)
        elif dataset == 'place365':
            valid = datasets.Places365(root='/mnt/Data/place365', split = 'val', small = True, download=False, transform=test_transform)
        elif dataset == 'dtd':
            valid = datasets.DTD(root='/mnt/Data/dtd', split = 'test', download=False, transform=test_transform)
        else:
            print('Wrong dataset name!')
            sys.exit(0)

        valid_dl = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif data_mode == 'test':
        if dataset == 'cifar10':
            valid = datasets.CIFAR10(root=r'F:\QML\Papers\CODES\FQN_HEDL\Data\CIFAR10', train=False, download=True, transform=test_transform)
        elif dataset == 'cifar100':
            valid = datasets.CIFAR100(root='/home/miyu/桌面/shift-wiseConv-master/cifar-100-python', train=False, download=False, transform=test_transform)
        elif dataset == 'cifar100s':
            valid = datasets.CIFAR100(root='/home/miyu/桌面/shift-wiseConv-master/cifar-100-python', train=False, download=False, transform=test_transform)
        elif dataset == 'flower':
            valid = FlowerDataset(transform=test_transform,data_mode='test')
        elif dataset == 'cub':
            valid = CubDataset(transform=test_transform,data_mode='valid')
        elif dataset == 'SVHN':
            valid = datasets.SVHN(root='F:\QML\Papers\CODES\hedl\datasets\SVHN', split='test', download=True, transform=test_transform)
        else:
            print('Wrong dataset name!')
            sys.exit(0)
        
        valid_dl = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('dataset:',dataset,'   mode:',data_mode,'   size:',len(valid))
    return valid_dl