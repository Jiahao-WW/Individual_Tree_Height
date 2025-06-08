# -*- encoding: utf-8 -*-
'''
file       :processfun.py
Description: 预处理函数
Date       :2023/11/08 14:47:59
Author     :Jiahao
'''
import pandas as pd
import matplotlib.pyplot as plt  # plotting tools
import geopandas as gpd
import numpy as np
import rasterio
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import shape
from rasterio.features import geometry_mask
from shapely.geometry import MultiPolygon
import glob
from rasterio.windows import Window
from rasterio.merge import merge
import warnings
from tqdm import tqdm
import re 
import os
from scipy.interpolate import RectBivariateSpline
from skimage.measure import label, regionprops

os.environ['PROJ_LIB'] = 'C:/Users/45958/.conda/envs/deep/Library/share/proj'

class preprocessing:
    def __init__(self, tiffpath, shppath, outpath, clippath, tiffilename):

        #读取tiff
        with rasterio.open(tiffpath) as src:
            self.profile = src.profile  # 读取头文件
            self.transform = src.transform  # 获取空间变换信息
            self.crs = src.crs  # 获取坐标参考系统
        

        
        self.tiffpath = tiffpath
        self.shppath = shppath
        self.outpath = outpath
        self.clippath = clippath
        self.filename = tiffilename

# 读取 SHP 文件
    def shp2tif(self, write = False):
        
        shpdata = gpd.read_file(self.shppath)
        geoms = list(shpdata['geometry'])
        tifshape = (self.profile['height'],self.profile['width'])

        # 创建标签图像
        labels = rasterize(shapes=geoms, out_shape=tifshape, 
                           transform=self.transform, fill=0, all_touched=False)

        # 将shp区域赋值为1（其他区域已经赋值为0）
        labels[labels > 0] = 1

        # 保存为 TIFF 文件
        segfilepath = self.outpath + self.filename + '_seg.tif'

        if write:
            with rasterio.open(segfilepath, 'w', driver='GTiff', height=self.profile['height'], width=self.profile['width'],
                               count=1, dtype=np.uint8, crs=self.crs, transform=self.transform) as dst:
                dst.write(labels, 1)
            print('已输出分割图')

        return

    def BoundaryWeight(self, scale_polygon = 1.5, output_plot = False, write = False): 
        '''
        For each polygon, create a weighted boundary where the weights of shared/close boundaries is higher than weights of solitary boundaries.
        '''
        shpdata = gpd.read_file(self.shppath)
        if shpdata.empty:
            return gpd.GeoDataFrame()
        
        tempPolygonDf = fix_topo(shpdata)
        #剔除多个多边形区域
        tempPolygonDf['geometry'] = tempPolygonDf['geometry'].apply(lambda x: multipolygon_to_polygon(x) if x.geom_type == 'MultiPolygon' else x)


        tempPolygonDf.reset_index(drop=True, inplace=True)
        Crs = tempPolygonDf.geometry.crs

        new_c = []
        print('begin to process boundary')
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The indices of the two GeoSeries are different.")
            warnings.filterwarnings("ignore", message="CRS mismatch between the CRS of left geometries and the CRS of right geometries.")
           # for each polygon in the area, scale and compare with other polygons
            for i in tqdm(range(len(tempPolygonDf))):
                pol1 = gpd.GeoSeries(tempPolygonDf.iloc[i]['geometry'])
                sc = pol1.scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='center')
                
                scc = gpd.GeoDataFrame({'id': [None], 'geometry': [sc[0]]})
                scc = gpd.GeoDataFrame(pd.concat([scc]*len(tempPolygonDf), ignore_index=True))
                pol2 = tempPolygonDf[~tempPolygonDf.index.isin([i])]
                #pol2['geometry'] = pol2['geometry'].scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='center')
                te = pol2['geometry'].scale(xfact=scale_polygon, yfact=scale_polygon, zfact=scale_polygon, origin='center')
                
                # invalid intersection operations topo error
                try:
                    ints = scc.intersection(te)
                    for k in range(len(ints)):
                        if ints.iloc[k] is not None:
                            if not ints.iloc[k].is_empty:
                                new_c.append(ints.iloc[k])
                except Exception as e:
                    print('Intersection error:', str(e))

            new_c = gpd.GeoSeries(new_c)
            new_cc = gpd.GeoDataFrame({'geometry': new_c})

            # df may contain points other than polygons
            new_cc['type'] = new_cc['geometry'].type 
            new_cc = new_cc[new_cc['type'].isin(['Polygon', 'MultiPolygon'])]

            tempPolygonDf['type'] = tempPolygonDf['geometry'].type 
            tempPolygonDf = tempPolygonDf[tempPolygonDf['type'].isin(['Polygon', 'MultiPolygon'])]

            bounda = gpd.overlay(new_cc, tempPolygonDf, how='difference')
            bounda = bounda.set_crs(Crs, allow_override=True)

        #绘图
        if output_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
            print(bounda.geometry.crs)
            bounda.plot(ax=ax1, color='red')
            tempPolygonDf.plot(alpha=0.2, ax=ax1, color='b')
            print(tempPolygonDf.geometry.crs)
            bounda.plot(ax=ax2, color='red')
            plt.show()

        # change multipolygon to polygon
        bounda = bounda.explode(index_parts=True)
        bounda.reset_index(drop=True, inplace=True)
        
        #读取
        if write:
            boundfilepath = self.outpath + self.filename + '_bound.tif'
            geoms = list(bounda['geometry'])
            tifshape = (self.profile['height'],self.profile['width'])
            labels = rasterize(shapes=geoms, out_shape= tifshape, transform=self.transform, fill=0, all_touched=False)
            # 将mask区域赋值为1（其他区域已经赋值为0）
            labels[labels > 0] = 1

            with rasterio.open(boundfilepath, 'w', driver='GTiff', height=self.profile['height'], width=self.profile['width'],
                               count=1, dtype=np.uint8, crs=self.crs, transform=self.transform) as dst:
                dst.write(labels, 1)
            print('已输出边界权重')            

        return bounda
    
    def cal_density(self, kernelsize = 11, sigma = 3.5, write = False):
        #计算密度图

        shpdata = gpd.read_file(self.shppath)

        if shpdata.empty:
            return gpd.GeoDataFrame()
        
        withoutmulti = fix_topo(shpdata)
        #剔除多个多边形区域
        withoutmulti['geometry'] = withoutmulti['geometry'].apply(lambda x: multipolygon_to_polygon(x) if x.geom_type == 'MultiPolygon' else x)
        transform = self.profile['transform']
        polygons = []
        polygon_anns = []

        for i in withoutmulti.index:
            gm = withoutmulti.loc[i]['geometry']
            a, b = gm.centroid.x, gm.centroid.y
            row, col = rasterio.transform.rowcol(transform, a, b)
            zipped = list((row,col)) 

            polygons.append(zipped)
            
            c,d = zip(*list(gm.exterior.coords))
            row2, col2 = rasterio.transform.rowcol(transform, c, d)
            zipped2 = list(zip(row2,col2)) 
            polygon_anns.append(zipped2)

        shap = (self.profile['height'], self.profile['width'])

        density_map = densitymap(shap, polygons, kernel_size = kernelsize, sigma = sigma)

        if write:
            densfilepath = self.outpath + self.filename + '_dens.tif'
            with rasterio.open(densfilepath, 'w', driver='GTiff', height=self.profile['height'], width=self.profile['width'],
                               count=1, dtype=np.float32, crs=self.crs, transform=self.transform) as dst:
                dst.write(density_map, 1)
            print('已输出密度图')
        
        return density_map
    
    def clip(self, pattern, block_size = 256, overlap = 32):
        readpath = self.outpath + self.filename + pattern + '.tif'
        # 根据输入的字符串进行判断
        if pattern == "":
            readpath = self.tiffpath
            filepath = self.clippath + 'img/' + self.filename + pattern
        elif pattern == "_bound":
            filepath = self.clippath + 'label/bound/' + self.filename + pattern
        elif pattern == "_dens":
            filepath = self.clippath + 'label/dens/' + self.filename + pattern
        elif pattern == "_seg":
            filepath = self.clippath + 'label/seg/' + self.filename + pattern
        elif pattern == "_height":
            filepath = self.clippath + 'label/' + self.filename + pattern
        else:
            print("输入的文件夹名称不在选择范围内")
            return

        with rasterio.open(readpath) as src:
            block_tif(src, filepath, block_size, overlap)
        
        return
    
    
#裁剪tif
def block_tif(src, filepath, block_size, overlap):

    # 获取影像的基本信息
    width, height = src.width, src.height
    num_bands = src.count

    # 计算需要分块的数量
    num_blocks_x = width // (block_size - overlap)
    num_blocks_y = height // (block_size - overlap)

    # 开始分块
    for i in range(num_blocks_y + 1):  # 注意这里我们增加了1，以便处理最后一行
        for j in range(num_blocks_x + 1):  # 同样，这里也增加了1，以便处理最后一列
            # 计算当前块的位置
            start_x = j * (block_size - overlap)

            start_y = i * (block_size - overlap)

            # 对于最后一列和最后一行，需要特殊处理
            if i == num_blocks_y:
                start_y = height - block_size
            if j == num_blocks_x:
                start_x = width - block_size

            window = Window(start_x, start_y, block_size, block_size)

            # 读取当前块的数据
            block_data = src.read(window=window)

            # 将当前块的数据保存为新的tif文件
            with rasterio.open(filepath + f'_{i}_{j}.tif', 'w', driver='GTiff',
                            height=block_data.shape[1], width=block_data.shape[2],
                            count=num_bands, dtype=block_data.dtype,
                            crs=src.crs, transform=src.window_transform(window)) as dst:
                dst.write(block_data)
    return 

def writetif(src, shpdata, outputpath):
    profile = src.profile  # 读取头文件
    transform = src.transform  # 获取空间变换信息
    crs = src.crs  # 
    tifshape = (profile['height'],profile['width'])
    geoms = list(shpdata['geometry'])
    labels = rasterize(shapes=geoms, out_shape=tifshape.shape, 
                           transform=transform, fill=0, all_touched=False)

    # 将shp区域赋值为1（其他区域已经赋值为0）
    labels[labels > 0] = 1

        # 保存为 TIFF 文件
    with rasterio.open(outputpath, 'w', driver='GTiff', height=tifshape[0], width=tifshape[1],
                       count=1, dtype=np.uint8, crs=crs, transform=transform) as dst:
        dst.write(labels, 1)
    print('已输出')
    return


def merge_tif(filepath, pattern, outputpath):
    # 找到所有的块文件
    #filepath+'train_*.tif'
    block_files = glob.glob(filepath + pattern)

    # 读取所有的块文件
    src_files_to_mosaic = []
    for file in block_files:
        src = rasterio.open(file)
        src_files_to_mosaic.append(src)


    # 合并所有的块
    mosaic, out_trans = merge(src_files_to_mosaic)

    # 获取影像的基本信息
    out_meta = src.meta.copy()

    # 更新元数据
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    })
    src.close()

    # 保存合并后的影像
    with rasterio.open(outputpath, "w", **out_meta) as dest:
        dest.write(mosaic)


def guassian_kernel(size, sigma):
    rows=size[0] # mind that size must be odd number.
    cols=size[1]
    mean_x=int((rows-1)/2)
    mean_y=int((cols-1)/2)

    f=np.zeros(size)
    for x in range(0,rows):
        for y in range(0,cols):
            mean_x2=(x-mean_x)*(x-mean_x)
            mean_y2=(y-mean_y)*(y-mean_y)
            f[x,y]=(1.0/(2.0*np.pi*sigma*sigma))*np.exp((mean_x2+mean_y2)/(-2.0*sigma*sigma))
    return f

def densitymap(shape, points, kernel_size=11,sigma=3.5):

    [rows,cols]=shape[0], shape[1]
    d_map=np.zeros([rows,cols])

    f= guassian_kernel([kernel_size,kernel_size],sigma)
    #save gaussian kernel
    #np.save('kernel.npy', f)    

    normed_f=(1.0/f.sum())*f # normalization for each head.
    #save gaussian kernel
    #np.save('kernel_norm.npy', normed_f)    

    if len(points)==0:
        return d_map
    else:
        for p in points:
            r,c=int(p[0]),int(p[1])
            if r>=rows or c>=cols:
                # print('larger')
                continue
            
            for x in range(0,f.shape[0]):
                for y in range(0,f.shape[1]):
                    if x+((r+1)-int((f.shape[0]-1)/2))<0 or x+((r+1)-int((f.shape[0]-1)/2))>rows-1 \
                    or y+((c+1)-int((f.shape[1]-1)/2))<0 or y+((c+1)-int((f.shape[1]-1)/2))>cols-1:
                        continue
                        # print('skipping cases')
                    else:
                        d_map[x+((r+1)-int((f.shape[0]-1)/2)),y+((c+1)-int((f.shape[1]-1)/2))]+=normed_f[x,y]

    return d_map

def multipolygon_to_polygon(multipolygon):
    #过滤拓扑错误的多边形
    largest_area = 0
    largest_polygon = None

    for polygon in multipolygon.geoms:
        area = polygon.area
        if area > largest_area:
            largest_area = area
            largest_polygon = polygon
    
    if largest_polygon is not None:
        return largest_polygon
    else:
        return multipolygon

def file_sort(files):
    # 定义一个函数用于从文件名中提取数字作为排序键
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        else:
            return -1  # 如果文件名中没有数字，则返回一个较大的数，确保不会影响到排序顺序

    # 对文件列表按照数字从小到大进行排序
    sorted_files = sorted(files, key=extract_number)
    return sorted_files

def fix_topo(shp_data):

    fix_shp = shp_data.copy()

    invalid_geometries = shp_data[~shp_data.geometry.is_valid]

    # 修复自相交的几何对象
    for index, row in invalid_geometries.iterrows():

        if row['geometry'].is_valid:
            continue  # 跳过有效的几何对象
        
        # 通过缓冲区和求交操作来修复自相交的多边形
        buffered = row['geometry'].buffer(0)
        if buffered.is_empty:
            continue  # 跳过空几何对象
        
        # 如果是多部分几何对象，只保留最大的面积部分
        print('修复自相交 ', index)
        if isinstance(buffered, MultiPolygon):
            repaired_geometry = max(buffered, key=lambda a: a.area)
        else:
            repaired_geometry = buffered
        
        # 将修复后的几何对象替换原始数据中的几何对象
        fix_shp.at[index, 'geometry'] = repaired_geometry

    return fix_shp


def crop_tif(large_path, small_path, clipped_output):
    with rasterio.open(large_path) as src:
    # 获取大范围图像的坐标范围和分辨率
        large_transform = src.transform
        large_crs = src.crs
        # 读取小区域图像

        with rasterio.open(small_path) as ref:
            # 获取小区域图像的坐标范围和分辨率
            small_transform = ref.transform
            small_crs = ref.crs
            # 根据小区域图像的边界计算裁剪窗口
            window = src.window(*ref.bounds)
            width = np.ceil(window.width)
            height = np.ceil(window.height)
            new_window = ((window.row_off, window.row_off + height), (window.col_off, window.col_off + width))

            # 读取裁剪后的大范围图像数据
            clipped_data = src.read(window=new_window, boundless=False)

            # 创建裁剪后和重采样后的图像文件
            profile = src.profile

            profile.update(height=height, width=width, transform=src.window_transform(new_window), crs=small_crs)
            with rasterio.open(clipped_output, 'w', **profile) as dst:
                dst.write(clipped_data)

    return 

def down_sampling(org_path, aim_path, output_path):
    with rasterio.open(org_path) as src:
    # 读取原始数据
        data = src.read(1)
        with rasterio.open(aim_path) as ref:
            # 获取原始数据的坐标范围和分辨率
            x_range = np.linspace(src.transform[2], src.transform[2] + src.transform[0]*src.width, src.width) #经度
            y_range = np.linspace(src.transform[5] + src.transform[4]*src.height,src.transform[5], src.height) #纬度
            
            # 创建插值函数
            interp_func = RectBivariateSpline(x_range, y_range, data)
            
            # 定义新的坐标范围和分辨率
            new_x_range = np.linspace(x_range.min(), x_range.max(), ref.width)
            new_y_range = np.linspace(y_range.min(), y_range.max(), ref.height)
            
            # 在新的坐标范围上进行插值
            interpolated_data = interp_func(new_x_range, new_y_range)
            
            # 创建新的TIFF文件保存插值结果
        
            profile = src.profile
            profile.update(width=ref.width, height=ref.height, transform=ref.transform)
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(interpolated_data, 1)
        
    return 

def Removespots(img, threshold):
    # 进行连通区域分析，获取每个区域的属性
    labeled_image = label(img, connectivity=2)
    regions = regionprops(labeled_image)

    # 定义面积阈值，用于剔除较小的区域
    area_threshold = threshold  # 自定义阈值，根据需要进行调整

    # 迭代每个区域，根据面积进行筛选
    filtered_image = np.zeros_like(img)
    for region in regions:
        area = region.area  # 区域面积

        # 根据面积阈值筛选区域
        if area > area_threshold:
            coords = region.coords
            filtered_image[coords[:, 0], coords[:, 1]] = 1
    
    return filtered_image


