#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vasileios Vonikakis
@email: bbonik@gmail.com

This script gives a real-time demonstration of the facial expression analysis
model, and updates a simple set of graphs. It makes use of your camera and
analyses your face in real-time. DLIB landmarks are not very illumination
invariant, so this works better when there are no shadows on the face.
"""


import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
# from source.emotions_dlib import EmotionsDlib, plot_landmarks
import os
from utils import export_csv
import time
from pathlib import Path
from emonet.models import EmoNet
import torch
from utils import extract_features,mycopyfile



'''
1表示悲伤
2表示厌恶
3表示愤怒
4表示恐惧
5表示中性
6表示开心
'''
def get_filelist(detector,model,dir):
    types = []
    abled_types = ['厌恶', '开心', '悲伤', '中性', '快乐',  '愤怒', '恐惧']
    abled_types = ['01', '02', '03', '04', '05', '06']
    type_names = [ '悲伤', '厌恶', '愤怒','恐惧', '中性', '开心' ]
    people_name = None

    exsit_files = []

    for home, dirs, files in os.walk(dir):
        # for dir in dirs:
        #     print(dir)
        for root, dirs2, files2 in os.walk("./output/2022/normal_not/"):
            exsit_files = [f[:-9] for f in files2]

        #获取xlsx文件信息
        if len(files) > 0:
            label_xlsx = [filename for filename in files if filename.find('xlsx') > 0 and filename.find('~') < 0]
            if len(label_xlsx) > 0:
                label_filename = label_xlsx[-1]
                people_name= label_filename.split('.')[0]
                label_filename = os.path.join(home, label_filename)
                print(label_filename)
            else:
                people_name = home.split("/")[-1]
                abled_types = type_names

        for filename in files:
            # print(filename)

            # 如果是mp4文件，则获取sheet名称并进行情绪识别
            if filename.find('mp4') > 0:
                sheetname = filename.split('.mp4')[0].split('-')[-1][-2:]
                if sheetname not in abled_types:
                    continue
                types.append(sheetname)
                if sheetname == '快乐':
                    sheetname = '开心'
                # print(sheetname)
                fullname = os.path.join(home, filename)
                print(fullname)

                check_str = people_name + "-" + sheetname
                if check_str in exsit_files:
                    print("exsit")
                    continue

                if "01" != sheetname and '悲伤' != sheetname:
                    print("not type")
                    continue

                #进行情绪识别
                # label_filename, sheetname = 'E:/mer-database/陶老师的视频/5.17汇总/正常/HY-006/HY-006.xlsx', '开心'
                try:
                    # sheetname = type_names[int(sheetname)-1]
                    # true_labels = parse_labels_from_xls(label_filename, sheetname)
                    # save_filename = './output/' + fullname.split('\\')[-1].replace('.mp4','-识别结果.xlsx')
                    # save_filename = './output/normal/' + fullname.split('\\')[-1].replace('.mp4','-识别结果.csv')
                    # save_filename = './output/normal/' + people_name + '-'+ fullname.split('\\')[-1].replace('.mp4','-识别结果.csv')
                    save_filename = './output/2022/normal_not/' + people_name + '-'+ sheetname +'-识别结果.csv'
                    srcfile = "E:/mer-database/av/normal_not/" + people_name + '-'+ sheetname +'-识别结果.csv'
                    if os.path.isfile(srcfile) and not os.path.isfile(save_filename):  # 判断是否已经存在 且未复制
                        print("exist copyed")
                        dst_dir = './output/2022/normal_not/'
                        mycopyfile(srcfile, dst_dir)
                    else:
                        start_time = time.time()
                        extract_features(detector,model,fullname,save_filename)
                        speed_time = time.time() - start_time
                        print("take time: {} ".format(round(speed_time,2)))
                except Exception as e:
                    print("error")
                    print(e)
                    continue
        print("#######file list#######")

def get_simple_file(detector,model,path,people_name,type):
    try:

        fullname = os.path.join(path, people_name + "/" + type + ".mp4")
        # save_filename = './output/normal/' + fullname.split('\\')[-1].replace('.mp4','-识别结果.csv')
        save_filename = './output/normal/' + people_name + '-' + type + '-识别结果.csv'
        # save_filename = './output/normal_not/' + people_name + '-'+ fullname.split('\\')[-1].replace('.mp4','-识别结果.csv')
        start_time = time.time()
        extract_features(detector, model, fullname, save_filename)
        speed_time = time.time() - start_time
        print("take time: {} ".format(round(speed_time, 2)))
    except Exception as e:
        print("error")
        print(e)

if __name__=="__main__":

    filename = 'G:/mer-data/第三次（70正常+70痴呆）/正常/MM-006/06.mp4'
    detector = dlib.get_frontal_face_detector()
    n_expression = 8
    device = 'cuda:0'
    # Loading the model
    state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')

    print(f'Loading the model from {state_dict_path}.')
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net = EmoNet(n_expression=n_expression).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    testPath = "G:/mer-data/第三次（70正常+70痴呆）/正常/"
    testPath = "D:/mer-data/2021-09/正常/"
    testPath = "D:/mer-data/2021-09/痴呆/"
    testPath = "G:/第三次（70正常+70痴呆）/痴呆/"
    # testPath = "G:/第一次（27正常+39痴呆）/痴呆/TK-002/"
    # get_filelist(detector,net,testPath)
    # testPath = "G:/第一次（27正常+39痴呆）/痴呆/"
    # get_filelist(detector,net,testPath)
    # testPath = "G:/第二次（30正常+15痴呆）/痴呆/"
    # get_filelist(detector,net,testPath)
    # testPath = "G:/第三次（70正常+70痴呆）/痴呆/"
    # get_filelist(detector,net,testPath)
    testPath = "G:/计算机老师/第一次（12正常+15轻度认知障碍+39痴呆）2021.12.22/痴呆/"
    get_filelist(detector, net, testPath)
    testPath = "G:/计算机老师/第二次（19正常+5轻度认知障碍+21痴呆）2021.12.22/痴呆/"
    get_filelist(detector, net, testPath)
    testPath = "G:/计算机老师/第三次（26正常+42轻度认知障碍+72痴呆）2021.12.22/痴呆/"
    get_filelist(detector, net, testPath)
    testPath = "G:/计算机老师/第四次（39正常+36轻度认知障碍+6痴呆）2021.12.22/痴呆/"
    get_filelist(detector, net, testPath)
    # path, people_name, type = "G:/第一次（27正常+39痴呆）/正常/",'HQ-001','02'
    # get_simple_file(detector, net, path, people_name, type)
