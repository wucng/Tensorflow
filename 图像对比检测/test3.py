#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
图像变化检测
同一地区不同时间的图像进行逐像素作差，检测变化区域
"""

# import cv2
from scipy import ndimage
import gdal
import numpy as np
import os.path as path
# from sklearn import preprocessing
from datetime import datetime
import sys
# from six.moves import xrange


start_time=datetime.now()
# 为了支持中文路径，请添加下面这句代码
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")

gdal.AllRegister() #注册驱动
# ogr.RegisterAll()


img1_path=r'E:\06\L15-3345E-2312N.tif'
img2_path=r'E:\08\L15-3345E-2312N.tif'
# img1_path=sys.argv[1]
# img2_path=sys.argv[2]

img_name=img2_path.split('/')[-1].split('.')[0] # Linux 为'/' ;windows 为'\\'

src_ds=gdal.Open(img1_path)
ibands=src_ds.RasterCount # 波段数
xcount=src_ds.RasterXSize # 宽度
ycount=src_ds.RasterYSize # 高度

assert ibands>=3 ,'波段数不足3'

data_type=np.float32
# 获取原始影像数据类型
# eDT=src_ds.GetRasterBand(1).DataType
#
# if eDT==1:
#     eDT=gdal.GDT_Byte
#     data_type=np.uint8
# elif eDT==2:
#     eDT=gdal.GDT_UInt16
#     data_type = np.uint16
# elif eDT==3:
#     eDT=gdal.GDT_Int16
#     data_type = np.int16
# elif eDT==4:
#     eDT=gdal.GDT_UInt32
#     data_type = np.uint32
# elif eDT==5:
#     eDT=gdal.GDT_Int32
#     data_type = np.int32
srcband = src_ds.GetRasterBand(1)
R1 = srcband.ReadAsArray(0, 0, xcount, ycount).astype(data_type)

srcband = src_ds.GetRasterBand(2)
G1 = srcband.ReadAsArray(0, 0, xcount, ycount).astype(data_type)

srcband = src_ds.GetRasterBand(3)
B1 = srcband.ReadAsArray(0, 0, xcount, ycount).astype(data_type)

src_ds=None
src_ds = gdal.Open(img2_path)
ibands = src_ds.RasterCount  # 波段数
xcount = src_ds.RasterXSize  # 宽度
ycount = src_ds.RasterYSize  # 高度

geoTrans = src_ds.GetGeoTransform() # 获取地理参考6参数
srcPro=src_ds.GetProjection() # 获取坐标引用

assert ibands>=3 ,'波段数不足3'

srcband = src_ds.GetRasterBand(1)
R2 = srcband.ReadAsArray(0, 0, xcount, ycount).astype(data_type)

srcband = src_ds.GetRasterBand(2)
G2 = srcband.ReadAsArray(0, 0, xcount, ycount).astype(data_type)

srcband = src_ds.GetRasterBand(3)
B2 = srcband.ReadAsArray(0, 0, xcount, ycount).astype(data_type)

# 每张图各自像素作逐差
img1=abs(R1-G1)+abs(R1-B1)+abs(G1-B1)


img2=abs(R2-G2)+abs(R2-B2)+abs(G2-B2)

# 使用sklearn 进行数据缩放
# min_max_scaler=preprocessing.MinMaxScaler().fit(img1)
# img1=min_max_scaler.fit_transform(img1)
# img2=min_max_scaler.fit_transform(img2)

# 使用scipy进行数据缩放
img1=(img1-np.min(img1,0))/(np.max(img1,0)-np.min(img1,0)+0.001)
img2=(img2-np.min(img2,0))/(np.max(img2,0)-np.min(img2,0)+0.001)


img=abs(img1-img2)

img=img-np.mean(img,0)*1.1+np.var(img,0)*2 # 减去阈值

# 将小于0 的像素值置0
# python3.5~
# def change_pixel(img):
#     for i in range(xcount):
#         for j in range(ycount):
#             if img[i, j] < 0:
#                 img[i, j] = 0
# map(change_pixel,img)

# python 2.7
# def change_piexl(img,i,j):
#     if img[i,j]<0:
#         img[i,j]=0
# [change_piexl(img,i,j) for i in range(img.shape[0]) for j in range(img.shape[1])]

# 将小于0的重置为0
img=np.maximum(img,0)

# img=(img+abs(img))/2
# img=np.where(img>0,img,0)

# img_1=img.copy()
# img_1[img_1<0]=0
# img=img_1

# opencv 作中值滤波
# img = cv2.medianBlur(img,13)  # 速度快

# 中值滤镜更好地保留边缘
img = ndimage.median_filter(img, 5)


# 如果原始影像像素值为0，即没有数据不做检测，将img对应的位置像素值设为0
mask1=(R1!=0).astype(np.float32)
mask2=(R2!=0).astype(np.float32)
img=img*mask1*mask2

# 输出结果图像
module_path = path.dirname(__file__) # 返回脚本文件所在的工作目录
raster_fn =path.join(module_path,img_name+'_mask.tiff')  # 存放掩膜影像

# raster_fn=path.join(sys.argv[3],img_name+'_mask.tiff')

target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, xcount, ycount, 1, gdal.GDT_Byte)
target_ds.SetGeoTransform(geoTrans) # 设置掩膜的地理参考
target_ds.SetProjection(srcPro) # 设置掩膜坐标引用

target_ds.GetRasterBand(1).WriteArray(img, 0, 0)
# target_ds.GetRasterBand(1).WriteRaster(0,0,xcount,ycount,img.tobytes())
target_ds.FlushCache()

target_ds=None
src_ds=None
end_time=datetime.now()
print((end_time-start_time).total_seconds())
