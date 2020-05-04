# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:59:35 2020

@author: nemow
"""

import os
import sys
import argparse
from datetime import datetime

from torch.nn import functional as F
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




def eval_training():
    net.eval()


    val_loss = 0.0 # cost function error
    correct = 0.0
    a=0
    for (images, labels) in val_loader:
        a+=1
        images = Variable(images)
        labels = Variable(labels)
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = net(images)
            outputs = F.softmax(outputs)
        loss = loss_function(outputs, labels)
        val_loss += loss.item()
        prob, preds = outputs.max(1)
        pred_lst.append(preds[0].item())
        prob_lst.append(prob[0].item())


        print('{}of 14648'.format(a))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-conti', type=int, default=1, help='continue?')
    parser.add_argument('-chkpoint', type=str, default='drive', help='checkpoint path')

    args = parser.parse_args()

    net = xception()
    if args.gpu:
        net = net.cuda()    
        
    #data preprocessing:

    
    val_loader = get_dataloader(
        settings.TRAIN_MEAN,
        settings.TRAIN_STD,
        c_path = 'drive/My Drive/data/test/test_data.csv',
        p_path = 'drive/My Drive/data/test/',
        num_workers=args.w,
        batch_size=args.b,
        shuffle=False
    )
    
    start_epoch=0
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.StepLR(optimizer,50,gamma=0.2,last_epoch=-1) #learning rate decay
    if args.conti == 1:
        checkpoint = torch.load(args.chkpoint)
        net.load_state_dict(checkpoint)
        train_scheduler = optim.lr_scheduler.StepLR(optimizer,50,gamma=0.2,last_epoch=start_epoch)


    #create checkpoint folder to save model

    pred_lst = []
    prob_lst = []
    best_acc = 0.0
    eval_training()
    dataf = pd.read_csv('test_data.csv')
    pic_lst = dataf['image_name']
    vid_lst = dataf['video_name']
    label_lst = dataf['class_label']

    new_vid_lst = []
    new_label_lst = []
    new_prob0_lst = []
    new_prob1_lst = []



    temp = vid_lst[0]
    count = 0
    prob0 = 0
    prob1 = 0
    for j in range(len(label_lst)-1):
        if vid_lst[j+1] == vid_lst[j]:
            if pred_lst[j] == 0:
                prob0 += prob_lst[j]
                prob1 += (1-prob_lst[j])
            else:
                prob1 += prob_lst[j]
                prob0 += (1-prob_lst[j])
        else:
            if pred_lst[j] == 0:
                prob0 += prob_lst[j]
                prob1 += (1-prob_lst[j])
            else:
                prob1 += prob_lst[j]
                prob0 += (1-prob_lst[j])
            new_prob1_lst.append(prob1)
            new_prob0_lst.append(prob0)
            new_label_lst.append(label_lst[j])
            new_vid_lst.append(vid_lst[j])
            prob1 = 0
            prob0 = 0
    new_prob1_lst.append(prob1+(1-prob_lst[14647]))
    new_prob0_lst.append(prob1+prob_lst[14647])
    new_label_lst.append(label_lst[14647])
    new_vid_lst.append(vid_lst[14647])

    dic = {'video_name':new_vid_lst, 
          'class_label':new_label_lst,
          'real_probability':new_prob0_lst,
          'fake_probability':new_prob1_lst}
    wr_dic = pd.DataFrame(dic)
    wr_dic.to_csv('test_result.csv')


















