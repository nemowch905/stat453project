# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:33:11 2020

@author: nemow
"""
import sys
import os.path as osp
import os
import argparse
import pandas as pd

def file_name_new(file_dir):
    save_list = []
    for root, dirs, files in os.walk(file_dir):
        for ff in files:
            if ff[-1]=="g":
                save_list.append(osp.join(ff)) #当前目录路径  
    return(save_list)

parser = argparse.ArgumentParser(description='image_augmentation')
parser.add_argument('--file_dir', default='aug_pic', type=str, help='aug file')
args = parser.parse_args()
if __name__=='__main__':
    pic_lst = file_name_new(args.file_dir)
    label_lst = []
    for i in range(len(pic_lst)):
        label_lst.append(0)
    
    dic = {'image_name':pic_lst, 
           'class_label':label_lst}
    wr_dic = pd.DataFrame(dic)
    wr_dic.to_csv(args.file_dir + '/' + '99_data.csv')