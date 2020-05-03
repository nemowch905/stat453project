# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:54:41 2020

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
            if ff[-1]=="v":
                save_list.append(osp.join(ff)) #当前目录路径  
    return(save_list)

pic_lst = []
label_lst = []
video_lst = []
dire = 'comb'
csv_lst = file_name_new(dire)
print(csv_lst)
dataf00 = pd.read_csv(dire + '\\' + '\\' + csv_lst[0])
pic_lst00 = dataf00['image_name']
label_lst00 = dataf00['class_label']
video_lst00 = dataf00['video_name']
for i in range(len(pic_lst00)):
    pic_lst.append(pic_lst00[i]+'.jpg')
    label_lst.append(label_lst00[i])
    video_lst.append(video_lst00[i])
dataf01 = pd.read_csv(dire + '\\' + '\\' + csv_lst[1])
pic_lst01 = dataf01['image_name']
label_lst01 = dataf01['class_label']
video_lst01 = dataf01['video_name']
for i in range(len(pic_lst01)):
    pic_lst.append(pic_lst01[i]+'.jpg')
    label_lst.append(label_lst01[i])
    video_lst.append(video_lst01[i])







dic = {'image_name':pic_lst, 
       'class_label':label_lst,
       'video_name':video_lst}
wr_dic = pd.DataFrame(dic)
wr_dic.to_csv(dire + '\\' + '\\' + 'test_data.csv')













