# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:09:16 2020

@author: nemow
"""

import sys
import os
import shutil
import os.path as osp
import cv2
import pdb
import argparse
import json
import numpy

print(data_video["abhggqdift.mp4"])

video1 = cv2.VideoCapture()
if not video1.open(r"C:\Users\nemow\Downloads\dfdc_train_part_0\aayrffkzxn.mp4"):
    print("can not open the video")
_, frame1 = video1.read()
matrix1 = numpy.asarray(frame1)
video2 = cv2.VideoCapture()
if not video2.open(r"C:\Users\nemow\Downloads\dfdc_train_part_0\abhggqdift.mp4"):
    print("can not open the video")
_, frame2 = video2.read()
matrix2 = numpy.asarray(frame2)

if not (matrix1==matrix2).all():
    print("success")

def teee(a,b):
    c = a+b
    print(c)
    return(c)

m = teee(1,2)

a = ['aaa','bbb']
b = [1,2]
c = list('a' = a,'b' = b)





















