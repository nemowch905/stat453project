# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:28:26 2020

@author: nemow
"""

import pandas as pd
import random
dataf = pd.read_csv('save_pic_24/24_data.csv')
dataf = dataf.sample(frac = 1).reset_index(drop = False)
lab = dataf['class_label']
img = dataf['image_name']
count = 1
label_lst = []
pic_lst = []
for i in range(len(lab)):
    print(img[i])
    if lab[i] == 0:
        label_lst.append(lab[i])
#        print(img[i])
        pic_lst.append(img[i]+'.jpg')
        count+=1
        if count >4000:
            break
count = 1
print(len(label_lst))
for i in range(len(lab)):
    if lab[i] == 1:
        label_lst.append(lab[i])
        pic_lst.append(img[i]+'.jpg')
        count+=1
        if count >4000:
            break

print(len(label_lst))
dic = {'image_name':pic_lst, 
       'class_label':label_lst}
wr_dic = pd.DataFrame(dic)
wr_dic.to_csv('val_data.csv')
