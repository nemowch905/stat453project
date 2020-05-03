# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:36:29 2020

@author: nemow
"""

import pandas as pd

with open("save.txt", "r") as f:
    data = f.readlines()
 
label_lst = []
pic_lst = []
for i,line in enumerate(data):
    pic_lst.append(line.strip())
    label_lst.append(0)
dic = {'image_name':pic_lst, 
       'class_label':label_lst}
wr_dic = pd.DataFrame(dic)
wr_dic.to_csv('99_data.csv')

