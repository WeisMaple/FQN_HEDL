
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


class FlowerDataset(Dataset):
    # 102 - 90 - 101
    def __init__(self, transform, size=224, data_mode='train', image_folder = '/mnt/Data/102flowers/jpg', label_mat = '/mnt/Data/102flowers/imagelabels.mat'):
        self.tfms = transform
        # 图片文件夹地址
        self.p = image_folder
        self.mat = label_mat
        # 图片id
        self.paths = list(os.listdir(self.p))
        # 图片标签
        self.labels = scipy.io.loadmat(self.mat)
        self.labels = np.array(self.labels['labels'][0]) - 1
        self.labels = list(self.labels)
        # 图片大小
        self.size = size
        self.data_mode = data_mode
        setid = scipy.io.loadmat('/mnt/Data/102flowers/setid.mat')

        valid = np.array(setid['valid'][0]) - 1
        train = np.array(setid['trnid'][0]) - 1
        test = np.array(setid['tstid'][0]) - 1

        if self.data_mode == 'train':
            self.paths = [i for j, i in enumerate(self.paths) if j in train]
            self.labels = [i for j, i in enumerate(self.labels) if j in train]
            print('training samples number is: ', len(self.paths))
        elif self.data_mode == 'valid':
            self.paths = [i for j, i in enumerate(self.paths) if j in valid]
            self.labels = [i for j, i in enumerate(self.labels) if j in valid]
            print('validation samples number is: ', len(self.paths))
        else:
            self.paths = [i for j, i in enumerate(self.paths) if j in test]
            self.labels = [i for j, i in enumerate(self.labels) if j in test]
            print('Test samples number is: ', len(self.paths))

    def __getitem__(self, idx):
        # Convert image to tensor and pre-process using transform
        img = PIL.Image.open(self.p+'/'+self.paths[idx]).convert('RGB')
        img = self.tfms(img)

        # Convert caption to tensor of word ids.
        target = torch.tensor(self.labels[idx], dtype=torch.int64)
        target = torch.nn.functional.one_hot(target, num_classes=102)

        # return pre-processed image and caption tensor
        return img, target

    def __len__(self):
        return len(self.paths)






