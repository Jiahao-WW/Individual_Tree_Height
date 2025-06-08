#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :main3_prediction.py
@说明        :prediction
@时间        :2024/09/16 12:47:07
@作者        :Jiahao W
'''

import numpy as np
import torch 
from dataload import normalize, normalize_out
from SAMUnet import SAMUNet
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from skimage.measure import label, regionprops
from osgeo import gdal
import math
from osgeo import osr
#import rasterio
from PIL import Image


def Removespots(img, threshold):
    # Perform connected region analysis to obtain the attributes of each region
    labeled_image = label(img, connectivity=2)
    regions = regionprops(labeled_image)

    # Define an area threshold for removing smaller areas
    area_threshold = threshold  

    # Create a new image to store the filtered regions
    # Initialize the filtered image with zeros (background)
    filtered_image = np.zeros_like(img)
    for region in regions:
        area = region.area  # 

        # if the area of the region is greater than the threshold, keep it
        if area > area_threshold:
            coords = region.coords
            filtered_image[coords[:, 0], coords[:, 1]] = 1
    
    return filtered_image

def writeTiff(input, number, outputname):

    #need to obtain the latitude and longitude of the lower left corner of the ima
    zoom = 18 #image level 19-1
    lon, lat = num2deg_TMS(number[0],number[1], zoom)    # 影像左上角经纬度
    # resolution
    resolution = 360 / (262144*256)
    lon_top = lon
    lat_top = lat + input.shape[0] * resolution

    # Determine the number of channels in the input array
    if input.ndim > 2:
        c = input.shape[2]
    else:
        c = 1

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(outputname, input.shape[1], input.shape[0], c, gdal.GDT_Byte)
    transform = [lon_top, resolution, 0, lat_top, 0, -resolution]
    dataset.SetGeoTransform(transform)

    # Set the data type
    #datatype = gdal.GDT_Byte
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())

    # Write the data to the dataset
    if c == 1:
        band = dataset.GetRasterBand(1)
        band.WriteArray(input)
    else:
        for i in range(c):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(input[:, :, i])

    # 关闭数据集
    dataset = None
    

def num2deg_TMS(x, y, zoom):

    n = math.pow(2, zoom)
    # Calculate the degree of each tile unit
    cell = 360 / n
    lon = x*cell-180
    lat = y*cell-90

    return [lon, lat]

def img_read(file_name):

    #leaving only the string after the last '/'
    file_parts = file_name.split('/')[-1] 

    file_name_only, extension = os.path.splitext(file_parts)

    # Split the file name section according to _
    x, y = map(int, file_name_only.split('_'))

    img = cv2.imread(file_name)

    result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return result, x, y

def calnumber(image, patch_size, overlap):
     
     c, m, n = image.shape
     numx = m// (patch_size-overlap) + 1
     numy = n// (patch_size-overlap) + 1

     counts = list(range(numx*numy))

     return counts, numx

class ImageDataset(Dataset):
    def __init__(self, image, patch_size=256, overlap=32):
        self.image = image  # image loaded
        self.patch_size = patch_size
        self.overlap = overlap
        self.counts, self.number = calnumber(self.image, self.patch_size, self.overlap)

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, idx):
        i = self.counts[idx] // self.number
        j = self.counts[idx] % self.number
        end_i = min(self.image.shape[1], i * (self.patch_size - self.overlap) + self.patch_size)
        end_j = min(self.image.shape[2], j * (self.patch_size - self.overlap) + self.patch_size)
        start_i = end_i - self.patch_size
        start_j = end_j - self.patch_size

        patch = self.image[:, start_i:end_i, start_j:end_j]

        tensor_patch = torch.Tensor(patch)

        return tensor_patch, start_i, start_j


if __name__ == '__main__':
     
    #initialize the model
    model = SAMUNet(
        encoder_name="resnext50_32x4d",  
        in_channels=3,  
        classes=2,  
        )

    model_path = r'D:\dataset\model\segment\samunet1019_nobound.pth'
    #samunet1019_nobound_best samunet0606_nobound_best

    # Load the model
    model.load_state_dict(torch.load(model_path))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.device_count()>1:
        print('GPUs')
        model = torch.nn.DataParallel(model)
    
    model.to(DEVICE)
    model.eval()

    # Set the patch size and overlap
    dx = 256
    dy = 256

    # Set the overlap size
    overlap = 32
    half = overlap//2 

    ###############prediction#####################

    # Read the image
    path = 'G:/googleearth/two_out/419_97/214144_81664.jpg'
    img, x, y = img_read(path)
    #data_set = gdal.Open(path)
    #img = data_set.ReadAsArray()
    img = np.array(img, np.float32)/255.0
    img = img.transpose(2, 0, 1)
    img = normalize_out(img)

    # pad_width to ensure the image is divisible by the patch size
    expanded_img = np.pad(img, pad_width=((0, 0), (32, 32), (32, 32)), mode='constant', constant_values=0)

    image_dataset = ImageDataset(image=expanded_img, patch_size=256, overlap=32)

    # DataLoader
    batch_size = 28
    data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the output image
    tt = 0
    c, m, n = image_dataset.image.shape
    output_image = np.zeros([m, n])

    for batch, start_i, start_j in data_loader:
        batch = batch.cuda()
        segs = model(batch)
        segs = torch.argmax(segs, axis=1)
        if tt % 10 ==0:
            print('finished:' + '%d'%tt)
        tt = tt +1
        segs = segs.cpu().data.numpy()
        for i in range(segs.shape[0]):

            output_image[start_i[i]+half:start_i[i]+dx-half, start_j[i]+half:start_j[i]+dy-half] = segs[i][half:dx-half, half:dy-half]

    output_image = output_image[32:m-32, 32:n-32]
    
    outputseg = Removespots(output_image, 4)  #remove small spots
    m,n = outputseg.shape

    # to jpg
    #
    image_name = f'{x}_{y}_seg.jpg'
    folder_path = 'E:/test'
    output_path = os.path.join(folder_path, image_name)
    #output_path = 'E:/test/' + f'{x}_{y}_seg.jpg'
    img = Image.fromarray(outputseg)

    # Ensure the output directory exists
    img.save(output_path)

    # to tiff
    #outputname = 'E:/test/' + f'{x}_{y}_seg.tif'
    #writeTiff(outputseg, [x,y], outputname)

    print('finish')