import os
import math
import random
import torch
import datetime, argparse, pprint
import fastai
import warnings
import sklearn
import json
import numpy as np
import pandas as pd
import evaluation
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
warnings.filterwarnings("ignore")

from dataloader import data_class, dataloader
from cub import CubDataset
from torch.utils.data import sampler
from torch.utils.data.sampler import SubsetRandomSampler
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, edl_HENN, edl_HENN_log, edl_HENN_mse,  get_belief, get_henn_fc
from evaluation import calculate_ood, calculate_socre, ODIN
from losses import relu_evidence
from fastai.vision import *
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from sklearn.metrics import classification_report
from models import SimpleCNN
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.distributions import Categorical
from torch.utils.data import ConcatDataset
from matplotlib.colors import ListedColormap
from torch.autograd import Variable
## Transforms
feat_dim = 512 # Hardcoded for resnet34

def validate(args, valid_dl, ood_dl, model):
    model.eval()
    total_uncertainty, total_label = [], []
    running_corrects, size_k = 0.0, 0
    with torch.no_grad():
        print ('Starting the In-Distribution Testing')
        for inputs, labels in tqdm(valid_dl):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            if args.dataset == 'cifar100':
                labels = torch.nn.functional.one_hot(labels, num_classes=100)
            if args.dataset == 'cifar10':
                labels = torch.nn.functional.one_hot(labels, num_classes=10)

            if args.loss == 'Softmax' or args.loss == 'dropout':
                features, outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                total_uncertainty.append(outputs)

            elif args.loss == 'edl_HENN' or args.loss == 'edl_HENN_log'  or args.loss == 'edl_HENN_mse' :
                features, outputs = model(inputs)
                sum_belief = get_henn_fc(features, model.get_weight(), args.w, args.num_classes, args.device)
                prob = sum_belief / torch.sum(sum_belief, dim=1, keepdim=True)
                _, preds = torch.max(prob, 1)
                
                total_uncertainty.append(args.w / torch.sum(sum_belief, dim=1, keepdim=True))
            
            else:
                features, outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                evidence = relu_evidence(outputs)
                alpha = evidence + 1

                total_uncertainty.append(args.num_classes / torch.sum(alpha, dim=1, keepdim=True))
            
            total_label.extend(torch.ones(inputs.size(0), dtype=torch.long))

            _, labels = torch.max(labels, 1)
            running_corrects += torch.sum(preds == labels.data)
            size_k += inputs.size(0)
            
        
        print ('Starting the OOD Testing')
        for inputs, labels in tqdm(ood_dl):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            
            # Pass the inputs through the CNN model.
            if args.loss == 'Softmax' or args.loss == 'dropout':
                features, outputs = model(inputs)
                
                total_uncertainty.append(outputs)
            
            elif args.loss == 'edl_HENN' or args.loss == 'edl_HENN_log'  or args.loss == 'edl_HENN_mse':
                features, outputs = model(inputs)
                sum_belief = get_henn_fc(features, model.get_weight(), args.w, args.num_classes, args.device)
                total_uncertainty.append(args.w / torch.sum(sum_belief, dim=1, keepdim=True))

            else:
                features, outputs = model(inputs)
                evidence = relu_evidence(outputs)
                alpha = evidence + 1
                
                total_uncertainty.append(args.num_classes / torch.sum(alpha, dim=1, keepdim=True))
            
            total_label.extend(torch.zeros(inputs.size(0), dtype=torch.long))

    total_uncertainty = torch.cat(total_uncertainty, 0)
    total_label = np.array(total_label)
    print(total_uncertainty.shape, len(total_label))

    return running_corrects / size_k, total_uncertainty, total_label
    

def main(args,options):
    # ========== 手动添加：写死正确的 checkpoint 路径 ==========
    args.checkpoint = r"F:\QML\Papers\CODES\hedl\code\runs_openset_cifar10\ResNet18\HEDL\acc\checkpoints\checkpoint.pth"
    # ========== 手动修改：固定 name，避免被覆盖 ==========
    args.name = "HEDL"
    # os.makedirs(os.path.join(args.output_dir,args.name,args.ood), exist_ok=True)
    
    # if args.name == 'TEST':
    #     args.name = str(args.loss)
    #else:
        #args.name = str(args.name) + '/' + str(args.ood)
    #args.checkpoint = ('runs_openset_' + args.dataset + '/' + str(args.model) + '/' + str(args.name) + '/acc/checkpoints/checkpoint.pth')
    
    print (args)
    print('Resume from checkpoint {}...'.format(args.checkpoint))
    

    use_gpu = torch.cuda.is_available()
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu': use_gpu,
            'checkpoint': args.checkpoint,
            'name': args.name
        }
    )
    args.num_classes = data_class(dataset=args.dataset)
    options.update(
        {
            'num_classes': args.num_classes
        }
    )
    if use_gpu:
        cudnn.benchmark = True
    else:
        print("Currently using CPU")
    
    valid_dl = dataloader(dataset=args.dataset,data_mode='test',batch_size=args.batch_size,num_workers=args.num_workers, image_size=args.image_size)
    ood_dl = dataloader(dataset=args.ood,data_mode='valid',batch_size=args.batch_size,num_workers=args.num_workers, image_size=args.image_size)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # load model
    print('Build and load model parameters...')
    args.output_dir = args.dataset + '/' + str(args.model) 
    print('total classes is {}'.format(args.num_classes))
    model = SimpleCNN(args.model,args.num_classes,loss=args.loss)
    model.to(args.device)
    model.load_state_dict(checkpoint['model'])
    print ("Loaded model, model:{} loss:{}".format(args.model,args.loss))
    os.makedirs(os.path.join(args.output_dir,args.name), exist_ok=True)
    
    ood_res = dict()
    print('Run validation...')
    ood_res['acc'], total_uncertainty, total_label = validate(args,valid_dl, ood_dl, model)
    ood_res['acc'] = ood_res['acc'].item()
    ood_res['err'] = 1.0 - ood_res['acc']
    print('out metrix:')
    
    if args.loss == 'edl_digamma_loss' or args.loss == 'edl_log_loss' or args.loss == 'edl_HENN':
        print('uncertainty ing...')
        ood_res['fpr95_uncertainty'],ood_res['auroc_uncertainty'],ood_res['auprc_uncertainty'] = calculate_ood(calculate_socre(total_uncertainty, 'uncertainty'), total_label)
    else:
        print('msp ing...')
        ood_res['fpr95_msp'],ood_res['auroc_msp'],ood_res['auprc_msp'] = calculate_ood(calculate_socre(total_uncertainty, 'msp'), total_label)
        print('entropy ing...')
        ood_res['fpr95_entropy'],ood_res['auroc_entropy'],ood_res['auprc_entropy'] = calculate_ood(calculate_socre(total_uncertainty, 'entropy'), total_label)
        print('energy ing...')
        ood_res['fpr95_energy'],ood_res['auroc_energy'],ood_res['auprc_energy'] = calculate_ood(calculate_socre(total_uncertainty, 'energy'), total_label)
    print(ood_res)

    import json
    with open(os.path.join(args.output_dir, args.name, 'ood_res.json'), 'w') as file:
        json.dump(ood_res, file)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description='Validate model with the test dataset')
    # data
    parser.add_argument('--data-dir', default='/mnt/mnt_data/Data/', help='data directory')
    parser.add_argument('--csv-file', default='/mnt/mnt_data/Data/', help='path to csv file')
    parser.add_argument('--checkpoint',type=str, default='test', help='path to checkpoint file')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='device {cuda:0, cpu}')
    parser.add_argument('--output-dir', default='TEST', help='where to save results')
    parser.add_argument('--name', default='TEST', help='where to save results')
    # data params
    parser.add_argument('--image-size', type=int, default=224, help='image size to the model')
    parser.add_argument('--num-workers', type=int, default=8, help='number of data loader workers')
    parser.add_argument('--batch-size', type=int, default=128, help='mini batch size')
    # other
    parser.add_argument('--debug', default=False, action='store_true', help='turn on debug mode')
    parser.add_argument('--debug-count', type=int, default=0, help='# of minibatchs for fast testing, 0 to disable')
    parser.add_argument('--loss',type=str, default='Softmax', help='Softmax/edl_digamma_loss/edl_log_loss/edl_mse_loss/edl_HENN')
    parser.add_argument('--model',type=str, default='ResNet18', help='ResNet34/ResNet18')
    parser.add_argument('--ood',type=str, default='SVHN', help='SVHN/dtd/place365')
    parser.add_argument('--dataset',type=str, default='cifar100', help='cifar10/cifar100/cub/flower')
    parser.add_argument('--temp', type=float, default=1.0, help="temp")
    parser.add_argument('--w', type=float, default=2.0, help="w")
    parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for RPL loss")
    options = vars(parser.parse_args())
    main(parser.parse_args(),options)