# -*- coding: utf-8 -*-
'''
对预测出的图像进行噪声去除，
选取一个3x3的模版，如果中间有像素值，而周围8个邻域都没有像素值
则去除掉中间这个像素值，已达到去除孤立噪声

注：由于做水位提取时，使用10x10的像素框，所以预测图像上一个一个矩形框 代表 10x10
因此，这里3x3的模版，其实应该是 30x30的像素框
'''

from osgeo import gdal, ogr
from osgeo.gdalconst import *
import numpy as np
import os
from os import path
# import gdalnumeric

# 为了支持中文路径，请添加下面这句代码
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")

gdal.AllRegister() #注册驱动
ogr.RegisterAll()

# module_path = path.dirname(__file__) # 返回脚本文件所在的工作目录

# srcImagePath=path.join(module_path,"34561.tif")

srcImagePath=r"F:\PL\2\34561.tiftest_mask.tiff"
srcDS=gdal.Open(srcImagePath,GA_ReadOnly)# 只读方式打开原始影像
geoTrans = srcDS.GetGeoTransform() # 获取地理参考6参数
srcPro=srcDS.GetProjection() # 获取坐标引用
srcXSize=srcDS.RasterXSize # 宽度
srcYSize=srcDS.RasterYSize # 高度
nbands=srcDS.RasterCount # 波段数

isize=10 # 制作10x10的样本

# target_ds = gdal.GetDriverByName('GTiff').CreateCopy(srcImagePath+'.tiff',srcDS) # 复制原始影像
target_ds = gdal.GetDriverByName('GTiff').Create(srcImagePath+'.tiff', srcXSize, srcYSize, 1, gdal.GDT_Byte)# 1表示1个波段，按原始影像大小
target_ds.SetGeoTransform(geoTrans) # 设置掩膜的地理参考
target_ds.SetProjection(srcPro) # 设置掩膜坐标引用
sBuf = srcDS.GetRasterBand(1).ReadAsArray(0,0,srcXSize,srcYSize,srcXSize,srcYSize).astype(np.uint8)# 读取所有像素
# dBuf=sBuf # 存储输出像素值

flagY = True
for i in range(0,srcYSize,isize):
    if not flagY: break
    # if m>100:break # 查看100张情况
    if i+isize*3>srcYSize-1:
        i=srcYSize-1-isize*3
        flagY=False

    flagX = True
    for k in range(0, srcXSize, isize):  # 如果步长取isize 由于存在黑边像素，会导致丢失一些图像
        if not flagX: break
        if k + isize*3 > srcXSize - 1:
            k = srcXSize - 1 - isize*3
            flagX = False

        # sBuf = srcDS.GetRasterBand(1).ReadAsArray(k, i, isize*3, isize*3, isize*3, isize*3).astype(
        #     np.uint8)  # 读取所有像素

        # if np.sum(sBuf[k+isize-1:k+2*isize-1,i+isize-1:i+2*isize-1])==255*isize*isize and \
        #                 np.sum(sBuf[k:k+3*isize-1,i:i+3*isize-1])==255*isize*isize:# 中间10x10 的像素都是白的(当时设置值为255)
        #                  # 只有中间10x10 的像素区域有像素值

        # if np.sum(sBuf[k+isize:k+2*isize,i+isize:i+2*isize])==np.sum(sBuf[k:k+3*isize,i:i+3*isize])\
        #         and np.sum(sBuf[k:k+3*isize,i:i+3*isize])!=0:
        #      sBuf[k:k + 3 * isize , i:i + 3 * isize]=np.zeros((isize*3,isize*3),np.uint8)  # 这种情况去除掉 ，全设置为0

        if np.sum(sBuf[i+isize:i+2*isize,k+isize:k+2*isize])==np.sum(sBuf[i:i+3*isize,k:k+3*isize])\
                and np.sum(sBuf[i:i+3*isize,k:k+3*isize])!=0:# 3x3邻域，中间像素不为0，周围全为0
            sBuf[i:i + 3 * isize, k:k + 3 * isize]=np.zeros((isize*3,isize*3),np.uint8)  # 这种情况去除掉 ，全设置为0

        if 2*np.sum(sBuf[i+isize:i+2*isize,k+isize:k+2*isize])==np.sum(sBuf[i:i+3*isize,k:k+3*isize])\
                and np.sum(sBuf[i:i+3*isize,k:k+3*isize])!=0: # 3x3邻域，中间像素不为0，周围只有1个不为0
            sBuf[i:i + 3 * isize, k:k + 3 * isize]=np.zeros((isize*3,isize*3),np.uint8)  # 这种情况去除掉 ，全设置为0

        if 3*np.sum(sBuf[i+isize:i+2*isize,k+isize:k+2*isize])==np.sum(sBuf[i:i+3*isize,k:k+3*isize])\
                and np.sum(sBuf[i:i+3*isize,k:k+3*isize])!=0:# 3x3邻域，中间像素不为0，周围只有2个不为0
            sBuf[i:i + 3 * isize, k:k + 3 * isize]=np.zeros((isize*3,isize*3),np.uint8)  # 这种情况去除掉 ，全设置为0

target_ds.GetRasterBand(1).WriteArray(sBuf, 0, 0)
target_ds.FlushCache()  # 将数据写入文件



