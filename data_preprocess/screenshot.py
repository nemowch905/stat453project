# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 08:29:00 2020

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
import pandas as pd

            
   
parser = argparse.ArgumentParser(description='screenshot')
parser.add_argument('--begin_index', default=-100, type=int, help='the first index')
parser.add_argument('--exf', default=8, type=int, help='the extract frequency, default is 8')
parser.add_argument('--exroot', default=None, type=str, help='extract root')
parser.add_argument('--path', default=None, type=str, help='images saving path')
parser.add_argument('--type', default='.mp4', type=str, help='video type, default is .mp4')
args = parser.parse_args()
def file_name_new(file_dir):
    save_list = []
    for root, dirs, files in os.walk(file_dir):
        for ff in files:
            if ff[-1]=="4":
                save_list.append(osp.join(ff)) #当前目录路径  
    return(save_list)
def extract_frames(video_path, dst_folder, prefix):
    # 主操作
    video = cv2.VideoCapture()
    if not video.open(video_path):
        print("can not open the video")
        exit(1)
    count = 1
    index = 1
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % EXTRACT_FREQUENCY == 0:
            save_path = "{}/{}{:>03d}.jpg".format(dst_folder, prefix, index)
            print("save_path::", save_path)
            cv2.imwrite(save_path, frame)
            index += 1
        
        count += 1
    video.release()
    return(index-1)
    # 打印出所提取帧的总数
    print("Video: {} \n Totally save {:d} pics".format(video_path, index-1))

def file_name(file_dir):
    save_list = []
    for root, dirs, files in os.walk(file_dir):
        for ff in files:
            save_list.append(osp.join(root,ff)) #当前目录路径  
    return(save_list)

if __name__=='__main__':
   # 全局变量
    VIDEO_DIR = args.exroot
    video_lst = file_name_new(VIDEO_DIR)
    fake_video_lst = []
    origin_video_lst = []
    open_file = VIDEO_DIR + "\\" + "\\" + "metadata.json"
    with open(open_file,'rb') as f:
        data_video = json.load(f)
        for i in range(len(video_lst)):
            if data_video[video_lst[i]]['label'] == "FAKE":
                fake_video_lst.append(video_lst[i])
                origin_video_lst.append(data_video[video_lst[i]]['original'])
    EXTRACT_root = args.path # 存放帧图片的根目录
    print(EXTRACT_root)
    EXTRACT_FREQUENCY = args.exf #帧提取频率
    if not osp.exists(EXTRACT_root):
        os.mkdir(EXTRACT_root)
    video_index = args.begin_index
    file_lst = file_name(VIDEO_DIR)
    print(video_lst)
    whole_video_lst = []
    pic_num = []
    label_lst = []
    for v_file in video_lst:
        date_file = VIDEO_DIR+"\\"+"\\"+v_file
        (_ , aa) = osp.splitext(date_file)
        if aa == args.type:
            if data_video[v_file]['label'] == "FAKE":
                video1 = cv2.VideoCapture()
                if not video1.open(date_file):
                    print("can not open the video")
                _, frame1 = video1.read()
                matrix1 = numpy.asarray(frame1)
                original_video = VIDEO_DIR + "\\" + "\\" + data_video[v_file]['original']
                video2 = cv2.VideoCapture()
                if not video2.open(original_video):
                    print("can not open the video")
                _, frame2 = video2.read()
                matrix2 = numpy.asarray(frame2)
                if not (matrix1==matrix2).all():
                    path = date_file
                    prefix = 'video_{:04d}_'.format(video_index)
                    video_index+=1
                    num1 = extract_frames(path, EXTRACT_root, prefix)
                    whole_video_lst.append(v_file)
                    pic_num.append(num1)
                    label_lst.append("FAKE")
            else:
                path = date_file
                prefix = 'video_{:04d}_'.format(video_index)
                video_index+=1
                num1 = extract_frames(path, EXTRACT_root, prefix)
                whole_video_lst.append(v_file)
                pic_num.append(num1)
                label_lst.append("REAL")
    dic = {'videoname':whole_video_lst,
           'pictures':pic_num, 
           'label':label_lst}
    wr_dic = pd.DataFrame(dic)
    wr_dic.to_csv(VIDEO_DIR + '\\' + '\\' + 'save_data.csv')



















