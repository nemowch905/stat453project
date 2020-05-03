# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:16:29 2020

@author: nemow
"""
# Importing necessary functions 
from keras.preprocessing.image import ImageDataGenerator,  array_to_img, img_to_array, load_img 
import sys
import os
import shutil
import os.path as osp
import cv2
import pdb
import argparse
import pandas as pd
# Initialising the ImageDataGenerator class. 
# We will pass in the augmentation parameters in the constructor.

def aug(file_name, prefix, save_dir):
    datagen = ImageDataGenerator( 
            rotation_range = 40, 
            shear_range = 0.2, 
            zoom_range = 0.2, 
            horizontal_flip = True, 
            brightness_range = (0.5, 1.5)) 
    img = load_img(file_name)  
    x = img_to_array(img) 
    x = x.reshape((1, ) + x.shape)   
    i = 0
    for batch in datagen.flow(x, batch_size = 1, 
                              save_to_dir = save_dir,  
                              save_prefix = prefix, save_format ='jpg'): 
        i += 1
        if i > 5:
           break;


parser = argparse.ArgumentParser(description='image_augmentation')
parser.add_argument('--file_dir', default='save_pic_0', type=str, help='picture file')
parser.add_argument('--csv_name', default='00_data.csv', type=str, help='csvfile')
parser.add_argument('--save_dir', default='aug_pic', type=str, help='save director')
args = parser.parse_args()

if __name__=='__main__':
    if not osp.exists(args.save_dir):
        os.mkdir(args.save_dir)
    csvpath = args.file_dir + '/' + args.csv_name
    dataf = pd.read_csv(csvpath)
    pic_lst = dataf['image_name']
    label_lst = dataf['class_label']
    for index, pic_name in enumerate(pic_lst):
        if label_lst[index] == 0:
            print(pic_name)
            imp_name = args.file_dir + '/' + pic_name
            aug(imp_name+'.jpg', pic_name[:-4], args.save_dir)
    