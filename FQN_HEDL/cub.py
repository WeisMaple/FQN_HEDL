
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
from tqdm import tqdm
import scipy.io
import numpy as np
import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
import torchvision
import PIL

from torch.utils.data import Dataset
from torchvision import transforms
from fastai import *
from fastai.vision import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CubDataset(Dataset):
    # 200类鸟类数据集
    def __init__(self, transform, size=224, data_mode='train', root = r'F:\QML\Papers\CODES\FQN_HEDL\Data'):
        self.tfms = transform
        # 图片文件夹地址
        self.p = root
        # 图片id
        self.paths = []
        img_txt_file = open(os.path.join(self.p, 'images.txt'))
        label_txt_file = open(os.path.join(self.p, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.p, 'train_test_split.txt'))
        for line in img_txt_file:
            # 最后一个字符为换行符
            self.paths.append(line[:-1].split(' ')[-1])
        # 图片标签
        self.labels = []
        for line in label_txt_file:
            self.labels.append(int(line[:-1].split(' ')[-1]) - 1)
        # 图片大小
        self.size = size
        self.data_mode = data_mode

        train, valid = [], []
        for line in train_val_file:
            if int(line[:-1].split(' ')[-1]) == 1:
                train.append(int(line[:-1].split(' ')[0])-1)
            else:
                valid.append(int(line[:-1].split(' ')[0])-1)

        if self.data_mode == 'train':
            self.paths = [i for j, i in enumerate(self.paths) if j in train]
            self.labels = [i for j, i in enumerate(self.labels) if j in train]
            # print('training samples number is: ', len(self.paths))
        elif self.data_mode == 'valid':
            self.paths = [i for j, i in enumerate(self.paths) if j in valid]
            self.labels = [i for j, i in enumerate(self.labels) if j in valid]
            # print('validation samples number is: ', len(self.paths))


    def __getitem__(self, idx):
        # Convert image to tensor and pre-process using transform
        img = PIL.Image.open(self.p+'/images/'+self.paths[idx]).convert('RGB')
        img = self.tfms(img)

        # Convert caption to tensor of word ids
        target = torch.tensor(self.labels[idx], dtype=torch.int64)
        target = torch.nn.functional.one_hot(target, num_classes=200)

        # return pre-processed image and caption tensor
        return img, target

    def __len__(self):
        return len(self.paths)








