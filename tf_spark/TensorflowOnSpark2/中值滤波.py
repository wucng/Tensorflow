# -*- coding: UTF-8 -*-
"""
中值滤波去噪
"""

import cv2
import numpy as np
# import matplotlib.pyplot as plt
import gdal, ogr

img_path='11_mask.tif'
denoise_img_path=img_path+'_denoising.tif'
img = cv2.imread(img_path,0) #直接读为灰度图像

blur = cv2.medianBlur(img,11)

cv2.imwrite(denoise_img_path,blur)

# 去噪后的图像加上坐标信息
srcDS = gdal.Open(img_path)
geoTrans = srcDS.GetGeoTransform()
srcPro = srcDS.GetProjection()

target_ds = gdal.Open(denoise_img_path,gdal.GA_Update)
target_ds.SetGeoTransform(geoTrans)
target_ds.SetProjection(srcPro)

target_ds.FlushCache()

target_ds = None



