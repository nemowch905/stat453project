# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:19:05 2020

@author: nemow
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable


from PIL import Image
from conf import settings
from utils import WarmUpLR
from model.xception import xception
import pandas as pd


class FakedetectDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.img_names = df['image_name'].values
        self.y = df['class_label'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
    
def get_dataloader(mean, std, c_path, p_path, batch_size=16, num_workers=0, shuffle=True):

    transform0 = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    dataset = FakedetectDataset(csv_path=c_path,
                                    img_dir=p_path,
                                    transform=transform0)
    data_loader = DataLoader(
        dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return data_loader

def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(train_loader):   


        images = Variable(images)
        labels = Variable(labels)
        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
#        with torch.no_grad():
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(train_loader.dataset)
        ))
        if epoch <= args.warm:
            warmup_scheduler.step()
        #update training loss for each iteration


    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]


def eval_training(epoch):
    net.eval()


    val_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in val_loader:
        images = Variable(images)
        labels = Variable(labels)
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = net(images)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    file_handle = open('drive/My Drive/data/acc/save_acc.txt',mode='a+')
    file_handle.write('Epoch: {} validation set: Average loss: {:.4f}, Accuracy: {:.4f} \n'.format(
        epoch,
        val_loss / len(val_loader.dataset),
        correct.float() / len(val_loader.dataset)
    ))
    file_handle.close()
    print('Epoch: {}validation set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        epoch,
        val_loss / len(val_loader.dataset),
        correct.float() / len(val_loader.dataset)
    ))
    print()

    return correct.float() / len(val_loader.dataset)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-conti', type=int, default=0, help='continue?')
    parser.add_argument('-chkpoint', type=str, default='drive', help='checkpoint path')

    args = parser.parse_args()

    net = xception()
    if args.gpu:
        net = net.cuda()    
        
    #data preprocessing:
    train_loader = get_dataloader(
        settings.TRAIN_MEAN,
        settings.TRAIN_STD,
        c_path = 'drive/My Drive/data/train/train_data.csv',
        p_path = 'drive/My Drive/data/train/',
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    val_loader = get_dataloader(
        settings.TRAIN_MEAN,
        settings.TRAIN_STD,
        c_path = 'drive/My Drive/data/val/val_data.csv',
        p_path = 'drive/My Drive/data/val/',
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    start_epoch=0
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.StepLR(optimizer,50,gamma=0.2,last_epoch=-1) #learning rate decay
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'Xception', settings.TIME_NOW)
    if args.conti == 1:
        checkpoint = torch.load(args.chkpoint)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        train_scheduler = optim.lr_scheduler.StepLR(optimizer,50,gamma=0.2,last_epoch=start_epoch)


    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(start_epoch+1, settings.EPOCH):


        train(epoch)


        acc = eval_training(epoch)
        if epoch > args.warm:
            train_scheduler.step(epoch)
        if epoch % 5 == 0:
            state = {'model':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, checkpoint_path.format(net='Xception', epoch=epoch, type='10times'))
        #start to save best performance model after learning rate decay to 0.01 
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net='Xception', epoch=epoch, type='best'))
            best_acc = acc
            continue



