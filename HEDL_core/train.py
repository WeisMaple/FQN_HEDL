import torch
import pandas as pd
import numpy as np
import os
import sys
import argparse
import fastai
import torch.nn as nn
import time, datetime
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models

from dataloader import data_class, dataloader
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, edl_HENN, edl_HENN_log, edl_HENN_mse, get_belief, get_unique, get_dir_num, get_henn_fc
from losses import relu_evidence
from torch.utils.data import DataLoader
# from schedulers import get_scheduler
from fastai import *
from fastai.vision import *
# from fastai.vision import get_transforms
from models import SimpleCNN
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

#Hardcoded parameters
feat_dim = 512 #Resnet34 features

def train_one_epoch(args, writer, model, criterion, data_loader, optimizer, epoch):
    model.train()
    count = 0
    for images, targets in tqdm(data_loader):
        count += 1
        images = images.to(args.device)
        targets = targets.to(args.device)
        if args.dataset == 'cifar100':
            targets = torch.nn.functional.one_hot(targets, num_classes=100)
        if args.dataset == 'cifar10':
            targets = torch.nn.functional.one_hot(targets, num_classes=10)
        targets = targets.float()
        
        
        # Pass the inputs through the CNN model.
        if args.loss == 'Softmax' or args.loss == 'dropout':
            features, outputs = model(images)
            loss = criterion(outputs, targets)
        elif args.loss == 'edl_HENN' or args.loss == 'edl_HENN_log'  or args.loss == 'edl_HENN_mse' or args.loss == 'edl_HENN_ce':
            
            features, outputs = model(images)
            loss = criterion(args.w, features, model.get_weight(), targets, args.epochs, args.num_classes, 10, args.device, outputs)
            # loss = criterion(args.w, features, model.get_weight(), targets, args.epochs, args.num_classes, 10, args.device)
        else:
            features, outputs = model(images)
            loss = criterion(outputs, targets, args.epochs, args.num_classes, 10, args.device)
        
        if writer is not None and count % args.log_steps == 1:
            writer.add_scalars('Loss/train', {'loss': loss}, epoch*len(data_loader)+count)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.debug == 1 and count >= 1:
            return loss

    return loss  

@torch.no_grad()
def evaluate(args, writer, model, criterion, data_loader, epoch):
    model.eval()
    count = 0
    size = 0
    running_loss = 0.0
    running_corrects = 0
    for images, targets in tqdm(data_loader):
        count += 1
        images = images.to(args.device)
        targets = targets.to(args.device)
        if args.dataset == 'cifar100':
            targets = torch.nn.functional.one_hot(targets, num_classes=100)
        if args.dataset == 'cifar10':
            targets = torch.nn.functional.one_hot(targets, num_classes=10)
        targets = targets.float()
        
        
        # Pass the inputs through the CNN model.
        if args.loss == 'Softmax' or args.loss == 'dropout':
            _, outputs = model(images)
            loss = criterion(outputs, targets)
            _, preds = torch.max(outputs, 1)
        elif args.loss == 'edl_HENN' or args.loss == 'edl_HENN_log'  or args.loss == 'edl_HENN_mse':
            
            features, outputs = model(images)
            loss = criterion(args.w, features, model.get_weight(), targets, args.epochs, args.num_classes, 10, args.device)
            sum_belief = get_henn_fc(features, model.get_weight(), args.w, args.num_classes, args.device)
            _,preds = torch.max(sum_belief,1)
            preds = preds.cuda()
        else:
            features, outputs = model(images)
            loss = criterion(outputs, targets, args.epochs, args.num_classes, 10, args.device)
            _, preds = torch.max(outputs, 1)
        
        # Calculate the batch loss.
        running_loss += loss.item() * images.size(0)
        targets = targets.data.max(1)[1]
        running_corrects += torch.sum(preds == targets)
        
        size += images.size(0)

        if args.debug == 1 and count >=1:
            break

    # end epoch
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.double() / size
    
    disp_str = 'Epoch {} Losses: {:.4f}'.format(epoch+1, epoch_loss)
    if writer is not None:
        writer.add_scalars('Loss/valid', {'loss': epoch_loss}, epoch)
        writer.add_scalars('Accuracy/valid', {'acc': epoch_acc}, epoch)
        writer.add_text('Log/valid', disp_str, epoch)
        
    return epoch_loss, epoch_acc


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description='Training on CIFAR dataset')
    # data
    parser.add_argument('--img-size', type=int, default=224, help='Training image size to be passed to the network')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='number of loader workers')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='device {cuda:0, cpu}')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--log-dir', default='', help='where to store results')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--log_steps', type=int, default=100, help='Logging at steps')
    parser.add_argument('--debug',type=int, default=0, help='Debug mode: 1')
    parser.add_argument('--scheduler',type=str, default='cosine_warm_restarts_warmup',help='Type of scheduler')
    parser.add_argument('--num-restarts',type=int,default=2, help='Number of restarts for scheduler')
    parser.add_argument('--checkpoint_path',type=str, default=None, help='Checkpoint path for resuming the training')
    parser.add_argument('--name',type=str, default=None, help='Checkpoint path name')
    parser.add_argument('--model',type=str, default='ResNet18', help='ResNet34/ResNet18')
    parser.add_argument('--loss',type=str, default='Softmax', help='Softmax/edl_digamma_loss/edl_log_loss/edl_mse_loss/edl_HENN')
    parser.add_argument('--dataset',type=str, default='cifar100', help='cifar10/cifar100/flower/cub')
    parser.add_argument('--temp', type=float, default=1.0, help="temp")
    parser.add_argument('--w', type=float, default=2.0, help="w")
    parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for RPL loss")
    args = parser.parse_args()

    print (args)
    options = vars(args)
    use_gpu = torch.cuda.is_available()
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu': use_gpu
        }
    )
    if use_gpu:
        cudnn.benchmark = True
    else:
        print("Currently using CPU")
    
    args.num_classes = data_class(dataset=args.dataset)
    train_dl = dataloader(dataset=args.dataset,data_mode='train',batch_size=args.batch_size,num_workers=args.num_workers, image_size=args.img_size)
    valid_dl = dataloader(dataset=args.dataset,data_mode='valid',batch_size=args.batch_size,num_workers=args.num_workers, image_size=args.img_size)
    # Initialize the model
    # model
    print('Build model')
    print("Using", torch.cuda.device_count(), "GPUs.")
    options.update(
        {
            'num_classes': args.num_classes
        }
    )
    print('total classes is {}'.format(args.num_classes))
    
    model = SimpleCNN(args.model,args.num_classes,args.loss)
    model = model.to(args.device)
    
    print ("Loaded model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    # Define the loss function.
    if args.loss == 'Softmax' or args.loss == 'dropout':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'edl_digamma_loss':
        criterion = edl_digamma_loss
    elif args.loss == 'edl_log_loss':
        criterion = edl_log_loss
    elif args.loss == 'edl_mse_loss':
        criterion = edl_mse_loss
    elif args.loss == 'edl_HENN':
        criterion = edl_HENN
    elif args.loss == 'edl_HENN_log':
        criterion = edl_HENN_log
    elif args.loss == 'edl_HENN_mse':
        criterion = edl_HENN_mse
    else:
        parser.error("--loss error")

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])

    if args.name is None:
        args.name = args.loss

    args.log_dir = 'runs_openset_' + args.dataset + '/' + str(args.model) + '/' + str(args.name)
    #Checkpoint saving for models best at Openset and best at ID classification
    writer = SummaryWriter(log_dir = args.log_dir)
    output_dir = Path(writer.log_dir)

    checkpoint_path_acc = Path(os.path.join(output_dir,'acc', 'checkpoints'))
    os.makedirs(checkpoint_path_acc, exist_ok=True)
    checkpoint_path_acc = checkpoint_path_acc / 'checkpoint.pth'

    checkpoint_path_loss = Path(os.path.join(output_dir,'loss', 'checkpoints'))
    os.makedirs(checkpoint_path_loss, exist_ok=True)
    checkpoint_path_loss = checkpoint_path_loss / 'checkpoint.pth'
    args.start_epoch = 0

    best_valid_acc, best_valid_loss = 0.0, 100
    best_monitor_loss = None
    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch', epoch)
        epoch_start_time = time.time()
        # scheduler.step(epoch)

        #Train one epoch
        print ('\nTrain: ')
        train_loss = train_one_epoch(args, writer, model, criterion, train_dl, optimizer, epoch)

        # evaluate
        print('\nEvaluate: ')
        valid_loss, valid_acc = evaluate(args, writer, model, criterion, valid_dl, epoch)
        checkpoint_paths_acc, checkpoint_paths_loss = [], []
        
        print ("\n Train loss:",train_loss.item(),"Valid loss:",valid_loss, "Valid acc: ", valid_acc)
        print('\ndataset =',args.dataset, '  loss = ', args.loss,'  best acc = ', best_valid_acc,'  name = ', args.name)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint_paths_loss.append(checkpoint_path_loss)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            checkpoint_paths_acc.append(checkpoint_path_acc)

        for cp in checkpoint_paths_acc:
            print('Save checkpoint {}'.format(cp))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
            }, cp)

        for cp in checkpoint_paths_loss:
            print('Save checkpoint {}'.format(cp))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
            }, cp)

        epoch_total_time = time.time() - epoch_start_time
        epoch_total_time_str = str(datetime.timedelta(seconds=int(epoch_total_time)))
        print('Epoch training time {}\n'.format(epoch_total_time_str))

    if writer is not None: writer.close()