#-*- coding:utf-8 -*-
import os

img_dir = r'F:\projects\OpenImages\OCR\train_data\train_img'
label_dir = r'F:\projects\OpenImages\OCR\train_data\train_label'
num_workers = 4
pretrained_weights = ''

anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

IMAGE_MEAN = [123.68, 116.779, 103.939]

# online hard example mining
OHEM = True
checkpoints_dir = './checkpoints'
