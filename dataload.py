#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :dataload.py
@说明        :
@时间        :2023/08/02 18:56:51
@作者        :Jiahao W
'''
from osgeo import gdal
import os
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import numpy as np
from torchvision import transforms
import random
import numpy as np
from skimage.transform import rotate
from skimage.exposure import rescale_intensity

#normalize
transform = transforms.Compose([
    transforms.ToTensor()])

def randomtransforms(image, label1, label2):
    # 
    image = image.transpose(1, 2, 0)

    if random.random() > 0.5:
        angle = np.random.randint(4) * 90
        image = rotate(image, angle).copy() # 1: Bi-linear (default)
        label1 = rotate(label1, angle, order=0).copy() # Nearest-neighbor
        label2 = rotate(label2, angle, order=0).copy() # Nearest-neighbor

    
    # Flip left and right
    if random.random() > 0.5:
        image = np.fliplr(image).copy()
        label1 = np.fliplr(label1).copy()
        label2 = np.fliplr(label2).copy()


    # upside down
    if random.random() > 0.5:
        image = np.flipud(image).copy()
        label1 = np.flipud(label1).copy()
        label2 = np.flipud(label2).copy()


    # brightness
    ratio=random.random()
    if  ratio>0.5:
        image = rescale_intensity(image, out_range=(0, ratio)).copy() #(0.5, 1)

    image = image.transpose(2, 0, 1)

    return image, label1, label2

def normalize(image, mean, std):
    # c,m,n
    img = np.zeros(image.shape, dtype='float32')
    if image.shape == (3, 256, 256):  # 
        for i in range(3):
            # 
            img[i] = (image[i] - mean[i]) / std[i]
    else:
        print('shape is wrong')
        exit()

    return img

def normalize_out(image):
    img = np.zeros(image.shape, dtype='float32')
    u = np.mean(image, axis=(1,2))
    sigma = np.std(image, axis=(1,2))
    img = (image - u[:,np.newaxis, np.newaxis])/ sigma[:,np.newaxis, np.newaxis]
    return img

def read_data(root_path, mode = 'train', pattern = ['seg','bound']):
    images = []
    segs = []
    bounds = []

 
    image_root = os.path.join(root_path, mode+'/', 'img/')
    gt_root = os.path.join(root_path, mode+'/', 'label/')
 

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        images.append(image_path)
        #label
        parts = image_name.split("_")  #
        #segment
        new_seg = "_".join([parts[0], pattern[0], parts[1], parts[2]])
        seg_path = os.path.join(gt_root, pattern[0] +'/',new_seg)
        segs.append(seg_path)
        #boundry
        new_bound = "_".join([parts[0], pattern[1], parts[1], parts[2]])
        bound_path = os.path.join(gt_root, pattern[1] +'/', new_bound)
        bounds.append(bound_path)


    return images, segs, bounds

def data_loader(img_path, seg_path, bound_path):
    #reading the image data
    ##################.npy reading#################

    img = np.load(img_path)
    img = img/255.0

    #mean = np.array([0.29621485, 0.3185807, 0.2866311])
    #std = np.array([0.19578443, 0.17140186, 0.16930631])
    #img = normalize(img, mean, std) #
    img = normalize_out(img) 

    seg = np.load(seg_path)
    seg = np.expand_dims(seg, axis=0)

    bounds = np.load(bound_path)
    bounds = np.expand_dims(bounds, axis=0)
    
    img, seg, bounds = randomtransforms(img, seg, bounds) # data augmentation

    return img, seg, bounds

def data_test_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)
    mask = np.expand_dims(mask, axis=2)

  
    img = np.array(img, np.float32) 
    mask = np.array(mask, np.float32)

    return img, mask

class Mydataset(data.Dataset):

    def __init__(self, rootpath, mode='train', pattern = ['seg','bound']):
        self.root = rootpath
        self.mode = mode
        self.images, self.segs, self.bounds = read_data(self.root, self.mode, pattern)

    def __getitem__(self, index):
        img, seg, bounds = data_loader(self.images[index], self.segs[index], self.bounds[index])
        img = torch.Tensor(img)
        seg = torch.Tensor(seg) #Classification label
        bounds = torch.Tensor(bounds) #boundary label
        return img, seg, bounds

    def __len__(self):

        assert len(self.images) == len(self.segs)
        
        return len(self.images)
