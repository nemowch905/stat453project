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
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR
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
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)
        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
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
        outputs = net(images)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('validation set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        val_loss / len(val_loader.dataset),
        correct.float() / len(val_loader.dataset)
    ))
    print()

    return correct.float() / len(val_loader.dataset)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = xception()
    if args.gpu:
        net = net.cuda()    
        
    #data preprocessing:
    train_loader = get_dataloader(
        settings.TRAIN_MEAN,
        settings.TRAIN_STD,
        c_path = settings.train_csv_path,
        p_path = settings.train_pic_path,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    val_loader = get_dataloader(
        settings.TRAIN_MEAN,
        settings.TRAIN_STD,
        c_path = settings.val_csv_path,
        p_path = settings.val_pic_path,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.s
    )
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)



    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(1, settings.EPOCH):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01 
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))


