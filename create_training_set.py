# -*- coding:utf-8 -*-
import os
import csv
import cv2
import math
import random
import numpy as np
from scipy.io import loadmat
import  get_density_map


seed = 95461354
random.seed(seed)
N = 9
dataset = 'A'

#测试
path_img = '/media/xwj/Data/DataSet/shanghai_tech/data/original/shanghaitech/part_A_final/train_data/images/'
train_path_img = '/media/xwj/Data/DataSet/shanghai_tech/data/play/train/img/'
train_path_den = '/media/xwj/Data/DataSet/shanghai_tech/data/play/train/den/'
val_path_img = '/media/xwj/Data/DataSet/shanghai_tech/data/play/val/img/'
val_path_den =  '/media/xwj/Data/DataSet/shanghai_tech/data/play/val/den/'
gt_path = '/media/xwj/Data/DataSet/shanghai_tech/part_A_final/train_data/ground_truth/'

for i in [train_path_img, train_path_den, val_path_img, val_path_den]:
    if not os.path.exists(i):
        os.makedirs(i)

if dataset == 'A':
    num_images = 300
else:
    num_images = 400

num_val = math.ceil(num_images*0.1)
indices = list(range(1, num_images+1))
random.shuffle(indices)

for idx in range(1, num_images+1):
    i = indices[idx-1]
    if idx % 10 == 0:
        print('Processing {}/{} files'.format(idx, num_images))
    image_info = loadmat(''.join((gt_path, 'GT_IMG_', str(i), '.mat')))['image_info']
    input_img_name = ''.join((path_img, 'IMG_',str(i), '.jpg'))

    im = cv2.imread(input_img_name, 0)
    h, w = im.shape
    wn2, hn2 = w / 8, h / 8
    wn2, hn2 = int(wn2 / 8) * 8, int(hn2 / 8) * 8   #1/8 to the original image size
    annPoints =  image_info[0][0][0][0][0] - 1
    if w <= wn2 * 2:
        im = cv2.resize(im, [h, wn2*2+1], interpolation=cv2.INTER_LANCZOS4)
        annPoints[:, 0] = annPoints[:, 0] * 2 * wn2 / w
    if h <= hn2 * 2:
        im = cv2.resize(im, [hn2*2+1, w], interpolation=cv2.INTER_LANCZOS4)
        annPoints[:, 1] = annPoints[:,1] * 2 * hn2 / h
    h, w = im.shape
    a_w, b_w = wn2 + 1, w - wn2  # 1/8+1,7/8
    a_h, b_h = hn2 + 1, h - hn2

    im_density = get_density_map.get_density_map_gaussian(im, annPoints)
    for j in range(1, N+1):
        r1 = random.random()            #0<r1<1
        r2 = random.random()
        x = math.floor((b_w - a_w) * r1 + a_w) #choose the sample center randomly between (1/8 + r1*7/8) image size
        y = math.floor((b_h - a_h) * r2 + a_h)
        x1, y1 = int(x - wn2), int(y - hn2)         #sample offset
        x2, y2 = int(x + wn2 - 1), int(y + hn2 - 1)
        im_sampled = im[y1-1:y2, x1-1:x2]       #sample the image as the size of offset
        im_density_sampled = im_density[y1-1:y2, x1-1:x2]           #sample the density map in the according area as the size of offset

        annPoints_sampled = annPoints[
            list(
                set(np.where(np.squeeze(annPoints[:,0]) > x1)[0].tolist()) &
                set(np.where(np.squeeze(annPoints[:,0]) < x2)[0].tolist()) &
                set(np.where(np.squeeze(annPoints[:,1]) > y1)[0].tolist()) &
                set(np.where(np.squeeze(annPoints[:,1]) < y2)[0].tolist())
            )
        ]       #get out the ground truth point that in the according offset area

        annPoints_sampled[:, 0] = annPoints_sampled[:, 0] - x1   #make the point start as the crop image
        annPoints_sampled[:, 1] = annPoints_sampled[:, 1] - y1
        img_idx = ''.join((str(i), '_',str(j)))

        #write the sample file to the disk
        if idx < num_val:
            cv2.imwrite(''.join([val_path_img, img_idx, '.jpg']), im_sampled)
            with open(''.join([val_path_den, img_idx, '.csv']), 'w') as fout:
                writer = csv.writer(fout)
                writer.writerows(im_density_sampled)
        else:
            cv2.imwrite(''.join([train_path_img, img_idx, '.jpg']), im_sampled)
            with open(''.join([train_path_den, img_idx, '.csv']), 'w') as fout:
                writer = csv.writer(fout)
                writer.writerows(im_density_sampled)