# -*- encoding: utf-8 -*-
'''
file       :main1_preprocess.py
Description:Generate samples based on the drawn samples tif and shp
Date       :2023/11/12 11:31:48
Author     :Jiahao
'''


from preprocess.processfun import preprocessing
from preprocess import processfun
import glob
import os

tiff_path = 'D:/tree3d/train2d/train2/image/'
shp_path = 'D:/tree3d/train2d/train2/label/'
out_path = 'D:/tree3d/train2d/train2/out/'
clip_path = 'D:/tree3d/train2d/train2/test/train/'

# get the tif files
tif_files = glob.glob(os.path.join(tiff_path, '*.tif'))

# Sort the file list by number from small to large
sorted_tif_files = processfun.file_sort(tif_files)


Pattern = ['','_bound','_dens','_seg']
for file in sorted_tif_files:
    tif_filename = os.path.splitext(os.path.basename(file))[0]
    shp_filepath = os.path.join(shp_path, f"{tif_filename}.shp")
    for pa in Pattern:
        datapre = preprocessing(file, shp_filepath, out_path, clip_path, tif_filename)
        datapre.clip(pattern = pa, block_size = 256, overlap = 32)

    print('finish',tif_filename)     

