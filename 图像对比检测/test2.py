#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
图像变化检测
同一地区不同时间的图像进行逐像素作差，检测变化区域
"""

from scipy import ndimage

try:
    from osgeo import gdal
except:
    import gdal
import numpy as np
import os.path as path
from datetime import datetime
import sys


start_time=datetime.now()
# 为了支持中文路径，请添加下面这句代码
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")

gdal.AllRegister() #注册驱动


assert len(sys.argv)>=4,"参数不足 使用参考:\n python test.py 图一路径 图二路径 输出结果路径"

img1_path=sys.argv[1]
img2_path=sys.argv[2]

img_name=img2_path.split('/')[-1].split('.')[0] # Linux 为'/' ;windows 为'\\'

src_ds1=gdal.Open(img1_path)
ibands1=src_ds1.RasterCount
xcount1=src_ds1.RasterXSize
ycount1=src_ds1.RasterYSize

geoTrans1 = src_ds1.GetGeoTransform()
srcPro1=src_ds1.GetProjection()

assert ibands1>=3 ,'%s波段数不足3'%img1_path

data_type=np.float32

srcband = src_ds1.GetRasterBand(1)
# R1 = srcband.ReadAsArray(0, 0, xcount1, ycount1).astype(data_type)
R1=srcband.ReadRaster(0,0,xcount1,ycount1)
R1=np.fromstring(R1,np.uint8)
R1=np.reshape(R1,[ycount1,xcount1]).astype(data_type)


srcband = src_ds1.GetRasterBand(2)
# G1 = srcband.ReadAsArray(0, 0, xcount1, ycount1).astype(data_type)
G1=srcband.ReadRaster(0,0,xcount1,ycount1)
G1=np.fromstring(G1,np.uint8)
G1=np.reshape(G1,[ycount1,xcount1]).astype(data_type)

srcband = src_ds1.GetRasterBand(3)
# B1 = srcband.ReadAsArray(0, 0, xcount1, ycount1).astype(data_type)
B1=srcband.ReadRaster(0,0,xcount1,ycount1)
B1=np.fromstring(R1,np.uint8)
B1=np.reshape(R1,[ycount1,xcount1]).astype(data_type)

src_ds1=None
src_ds2 = gdal.Open(img2_path)
ibands2 = src_ds2.RasterCount
xcount2 = src_ds2.RasterXSize
ycount2 = src_ds2.RasterYSize

geoTrans2 = src_ds2.GetGeoTransform()
srcPro2=src_ds2.GetProjection()

assert ibands2>=3 ,'%s波段数不足3'%img2_path

assert ibands1==ibands2 ,'两张图像波段不一致'
assert xcount1==xcount2 ,'两张图像宽不一致'
assert ycount1==ycount2,'两张图像高不一致'
assert geoTrans1==geoTrans2,'两张图像空间参考不一致'
assert srcPro1==srcPro2,'两张图像投影不一致'

srcband = src_ds2.GetRasterBand(1)
# R2 = srcband.ReadAsArray(0, 0, xcount2, ycount2).astype(data_type)
R2=srcband.ReadRaster(0,0,xcount2,ycount2)
R2=np.fromstring(R2,np.uint8)
R2=np.reshape(R2,[ycount2,xcount2]).astype(data_type)

srcband = src_ds2.GetRasterBand(2)
# G2 = srcband.ReadAsArray(0, 0, xcount2, ycount2).astype(data_type)
G2=srcband.ReadRaster(0,0,xcount2,ycount2)
G2=np.fromstring(G2,np.uint8)
G2=np.reshape(G2,[ycount2,xcount2]).astype(data_type)

srcband = src_ds2.GetRasterBand(3)
# B2 = srcband.ReadAsArray(0, 0, xcount2, ycount2).astype(data_type)
B2=srcband.ReadRaster(0,0,xcount2,ycount2)
B2=np.fromstring(B2,np.uint8)
B2=np.reshape(B2,[ycount2,xcount2]).astype(data_type)

# 每张图各自像素作逐差
img1=abs(R1-G1)+abs(R1-B1)+abs(G1-B1)

img2=abs(R2-G2)+abs(R2-B2)+abs(G2-B2)

# 像素缩放
img1=(img1-np.min(img1,0))/(np.max(img1,0)-np.min(img1,0)+0.001)
img2=(img2-np.min(img2,0))/(np.max(img2,0)-np.min(img2,0)+0.001)


img=abs(img1-img2)

img=img-np.mean(img,0)*1.1+np.var(img,0)*2 # 减去阈值

# 将小于0的重置为0
img=np.maximum(img,0)

# 中值滤镜更好地保留边缘
img = ndimage.median_filter(img, 5)

# 如果原始影像像素值为0，即没有数据不做检测，将img对应的位置像素值设为0
mask1=(R1!=0).astype(np.float32)
mask2=(R2!=0).astype(np.float32)
img=img*mask1*mask2

# 输出结果图像

raster_fn=path.join(sys.argv[3],img_name+'_mask.tif')

target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, xcount1, ycount1, 1, gdal.GDT_Byte)
target_ds.SetGeoTransform(geoTrans1) # 设置掩膜的地理参考
target_ds.SetProjection(srcPro1) # 设置掩膜坐标引用

target_ds.GetRasterBand(1).WriteArray(img, 0, 0)
# target_ds.GetRasterBand(1).WriteRaster(0,0,xcount1,ycount1,img.tobytes())
# [target_ds.GetRasterBand(1).WriteArray(img, 0, i) for i in ycount1]
target_ds.FlushCache()

target_ds2=None
src_ds=None
end_time=datetime.now()
print((end_time-start_time).total_seconds())
