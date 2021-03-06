#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
图像变化检测
同一地区不同时间的图像进行逐像素作差，检测变化区域
按文件夹处理
"""

try:
    from osgeo import gdal
except:
    import gdal
import gdalnumeric
from osgeo.gdalconst import *
from glob import glob
import os
import itertools
from scipy import ndimage
import numpy as np
import sys
from datetime import datetime


def get_tif_path(file1,file2):
   '''
   返回tif/tiff对应的路径
   :param file1: 第一个文件夹
   :param file2: 第二个文件夹
   :return: 返回做匹配图像的2张图像对应路径
   '''
   file1_path = []
   [file1_path.append(glob(os.path.join(file1, '*/' * i, '*.tif'))) for i in range(3)] # 只循环3级目录
   [file1_path.append(glob(os.path.join(file1, '*/' * i, '*.tiff'))) for i in range(3)]
   file1_path = list(itertools.chain.from_iterable(file1_path))  # 转成一维list

   file2_path = []
   [file2_path.append(glob(os.path.join(file2, '*/' * i, '*.tif'))) for i in range(3)]
   [file2_path.append(glob(os.path.join(file2, '*/' * i, '*.tiff'))) for i in range(3)]
   file2_path = list(itertools.chain.from_iterable(file2_path))  # 转成一维list

   # 将file1_path 与 file2_path一一对应
   file_path=[]
   for img in file1_path:
       img_name = img.split('/')[-1].split('.')[0]  # Linux 为'/' ;windows 为'\\'
       for img2 in file2_path:
          if img_name in img2:
             file_path.append([img,img2])
             file2_path.remove(img2)
             break

   return [file1_path,file_path]

def get_geoPosition(file_path):
   '''
   解析文件夹中的所有影像，获取最终的地理位置，最左上角地理坐标和最右下角地理坐标
   :param file_path: 输入文件的tif影像路径
   :return: 地理坐标和参考系参数以及合并的图像大小
   '''
   gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
   gdal.AllRegister()

   flag = True

   # 循环所有图像找到最左上角地理坐标和最右下角地理坐标
   for file in file_path:
      srcDS = gdal.Open(file, GA_ReadOnly)  # 只读方式打开原始影像
      geoTrans = srcDS.GetGeoTransform()  # 获取地理参考6参数
      # srcPro = srcDS.GetProjection()  # 获取坐标引用
      srcXSize = srcDS.RasterXSize  # 宽度
      srcYSize = srcDS.RasterYSize  # 高度

      # 原始影像的左上角坐标 (geoTrans[0],geoTrans[3])
      srcX=geoTrans[0]
      srcY=geoTrans[3]
      # 原始影像的右下角坐标（地理坐标）
      srcX2 = geoTrans[0] + srcXSize * geoTrans[1] + srcYSize * geoTrans[2]
      srcY2 = geoTrans[3] + srcXSize * geoTrans[4] + srcYSize * geoTrans[5]# Y方向的坐标 递减 geoTrans[5]是负数

      if flag:
         srcPro = srcDS.GetProjection()

         left_top_X = srcX
         left_top_Y = srcY
         right_bottom_X = srcX2
         right_bottom_Y = srcY2
         flag = False


      if left_top_X > srcX: left_top_X = srcX
      if left_top_Y < srcY: left_top_Y = srcY

      if right_bottom_X<srcX2:right_bottom_X=srcX2
      if right_bottom_Y>srcY2:right_bottom_Y=srcY2

   # Create a new geomatrix for the image
   geoTrans2 = list(geoTrans)
   # 重新设置左上角地理坐标

   geoTrans2[0] = left_top_X
   geoTrans2[3] = left_top_Y

   # 所有图像合起来的图像大小
   Xsize = np.ceil((right_bottom_X - left_top_X) / geoTrans[1])
   Ysize = np.ceil((right_bottom_Y - left_top_Y) / geoTrans[5])

   return [geoTrans2,srcPro,Xsize,Ysize]

def Image_contrast_detection(img1_path,img2_path,img_save):
   '''
   2张图像对比检测，返回最后的输出结果图(只能两张两张图像做对比)
   :param img1_path: 图一路径
   :param img2_path: 图二路径
   :param img_save: 输出结果保存路径
   :return: 文件夹1,2对应图像对比结果
   '''

   # 为了支持中文路径，请添加下面这句代码
   gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")

   gdal.AllRegister()  # 注册驱动

   # assert len(sys.argv) >= 4, "参数不足 使用参考:\n python test.py 图一路径 图二路径 输出结果路径"

   # img1_path = sys.argv[1]
   # img2_path = sys.argv[2]

   img_name = img2_path.split('/')[-1].split('.')[0]  # Linux 为'/' ;windows 为'\\'

   src_ds1 = gdal.Open(img1_path)
   ibands1 = src_ds1.RasterCount
   xcount1 = src_ds1.RasterXSize
   ycount1 = src_ds1.RasterYSize

   geoTrans1 = src_ds1.GetGeoTransform()
   srcPro1 = src_ds1.GetProjection()

   assert ibands1 >= 3, '%s波段数不足3' % img1_path

   data_type = np.float32

   srcband = src_ds1.GetRasterBand(1)
   # R1 = srcband.ReadAsArray(0, 0, xcount1, ycount1).astype(data_type)
   R1 = srcband.ReadRaster(0, 0, xcount1, ycount1)
   R1 = np.fromstring(R1, np.uint8)
   R1 = np.reshape(R1, [ycount1, xcount1]).astype(data_type)

   srcband = src_ds1.GetRasterBand(2)
   # G1 = srcband.ReadAsArray(0, 0, xcount1, ycount1).astype(data_type)
   G1 = srcband.ReadRaster(0, 0, xcount1, ycount1)
   G1 = np.fromstring(G1, np.uint8)
   G1 = np.reshape(G1, [ycount1, xcount1]).astype(data_type)

   srcband = src_ds1.GetRasterBand(3)
   # B1 = srcband.ReadAsArray(0, 0, xcount1, ycount1).astype(data_type)
   B1 = srcband.ReadRaster(0, 0, xcount1, ycount1)
   B1 = np.fromstring(R1, np.uint8)
   B1 = np.reshape(R1, [ycount1, xcount1]).astype(data_type)

   src_ds1 = None
   src_ds2 = gdal.Open(img2_path)
   ibands2 = src_ds2.RasterCount
   xcount2 = src_ds2.RasterXSize
   ycount2 = src_ds2.RasterYSize

   geoTrans2 = src_ds2.GetGeoTransform()
   srcPro2 = src_ds2.GetProjection()

   assert ibands2 >= 3, '%s波段数不足3' % img2_path

   assert ibands1 == ibands2, '两张图像波段不一致'
   assert xcount1 == xcount2, '两张图像宽不一致'
   assert ycount1 == ycount2, '两张图像高不一致'
   assert geoTrans1 == geoTrans2, '两张图像空间参考不一致'
   assert srcPro1 == srcPro2, '两张图像投影不一致'

   srcband = src_ds2.GetRasterBand(1)
   # R2 = srcband.ReadAsArray(0, 0, xcount2, ycount2).astype(data_type)
   R2 = srcband.ReadRaster(0, 0, xcount2, ycount2)
   R2 = np.fromstring(R2, np.uint8)
   R2 = np.reshape(R2, [ycount2, xcount2]).astype(data_type)

   srcband = src_ds2.GetRasterBand(2)
   # G2 = srcband.ReadAsArray(0, 0, xcount2, ycount2).astype(data_type)
   G2 = srcband.ReadRaster(0, 0, xcount2, ycount2)
   G2 = np.fromstring(G2, np.uint8)
   G2 = np.reshape(G2, [ycount2, xcount2]).astype(data_type)

   srcband = src_ds2.GetRasterBand(3)
   # B2 = srcband.ReadAsArray(0, 0, xcount2, ycount2).astype(data_type)
   B2 = srcband.ReadRaster(0, 0, xcount2, ycount2)
   B2 = np.fromstring(B2, np.uint8)
   B2 = np.reshape(B2, [ycount2, xcount2]).astype(data_type)

   # 每张图各自像素作逐差
   img1 = abs(R1 - G1) + abs(R1 - B1) + abs(G1 - B1)

   img2 = abs(R2 - G2) + abs(R2 - B2) + abs(G2 - B2)

   # 像素缩放
   img1 = (img1 - np.min(img1, 0)) / (np.max(img1, 0) - np.min(img1, 0) + 0.001)
   img2 = (img2 - np.min(img2, 0)) / (np.max(img2, 0) - np.min(img2, 0) + 0.001)

   img = abs(img1 - img2)

   img = img - np.mean(img, 0) * 1.1 + np.var(img, 0) * 2  # 减去阈值

   # 将小于0的重置为0
   img = np.maximum(img, 0)

   # 中值滤镜更好地保留边缘
   img = ndimage.median_filter(img, 5)

   # 如果原始影像像素值为0，即没有数据不做检测，将img对应的位置像素值设为0
   mask1 = (R1 != 0).astype(np.float32)
   mask2 = (R2 != 0).astype(np.float32)
   img = img * mask1 * mask2

   # 输出结果图像

   # raster_fn = path.join(sys.argv[3], img_name + '_mask.tif')

   raster_fn=path.join(img_save,img_name + '_mask.tif')

   target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, xcount1, ycount1, 1, gdal.GDT_Byte)
   target_ds.SetGeoTransform(geoTrans1)  # 设置掩膜的地理参考
   target_ds.SetProjection(srcPro1)  # 设置掩膜坐标引用

   target_ds.GetRasterBand(1).WriteArray(img, 0, 0)
   # target_ds.GetRasterBand(1).WriteRaster(0,0,xcount1,ycount1,img.tobytes())
   # [target_ds.GetRasterBand(1).WriteArray(img, 0, i) for i in ycount1]
   target_ds.FlushCache()

   target_ds = None
   src_ds = None

def Image_detection(img_path):
   '''
   返回对比检测的结果数组numpy,图像的地理坐标
   :param img_path: img_path[0]图一，img_path[1]图二
   :return: 
   '''
   data_type = np.float32

   src_ds1 = gdal.Open(img_path[0], GA_ReadOnly)
   ibands1 = src_ds1.RasterCount
   xcount1 = src_ds1.RasterXSize
   ycount1 = src_ds1.RasterYSize

   geoTrans1 = src_ds1.GetGeoTransform()
   srcPro1 = src_ds1.GetProjection()

   assert ibands1 >= 3, '%s波段数不足3' % img_path[0]
   srcband = src_ds1.GetRasterBand(1)
   R1 = srcband.ReadAsArray(0, 0, xcount1, ycount1).astype(data_type)

   srcband = src_ds1.GetRasterBand(2)
   G1 = srcband.ReadAsArray(0, 0, xcount1, ycount1).astype(data_type)

   srcband = src_ds1.GetRasterBand(3)
   B1 = srcband.ReadAsArray(0, 0, xcount1, ycount1).astype(data_type)
   src_ds1 = None

   src_ds2 = gdal.Open(img_path[1], GA_ReadOnly)
   ibands2 = src_ds2.RasterCount
   xcount2 = src_ds2.RasterXSize
   ycount2 = src_ds2.RasterYSize

   geoTrans2 = src_ds2.GetGeoTransform()
   srcPro2 = src_ds2.GetProjection()

   assert ibands2 >= 3, '%s波段数不足3' % img_path[1]

   assert ibands1 == ibands2, '两张图像波段不一致'
   assert xcount1 == xcount2, '两张图像宽不一致'
   assert ycount1 == ycount2, '两张图像高不一致'
   assert geoTrans1 == geoTrans2, '两张图像空间参考不一致'
   assert srcPro1 == srcPro2, '两张图像投影不一致'

   srcband = src_ds2.GetRasterBand(1)
   R2 = srcband.ReadAsArray(0, 0, xcount2, ycount2).astype(data_type)

   srcband = src_ds2.GetRasterBand(2)
   G2 = srcband.ReadAsArray(0, 0, xcount2, ycount2).astype(data_type)

   srcband = src_ds2.GetRasterBand(3)
   B2 = srcband.ReadAsArray(0, 0, xcount2, ycount2).astype(data_type)

   src_ds2 = None
   # 每张图各自像素作逐差
   img1 = abs(R1 - G1) + abs(R1 - B1) + abs(G1 - B1)

   img2 = abs(R2 - G2) + abs(R2 - B2) + abs(G2 - B2)

   # 像素缩放
   img1 = (img1 - np.min(img1, 0)) / (np.max(img1, 0) - np.min(img1, 0) + 0.001)
   img2 = (img2 - np.min(img2, 0)) / (np.max(img2, 0) - np.min(img2, 0) + 0.001)

   img = abs(img1 - img2)

   img = img - np.mean(img, 0) * 1.1 + np.var(img, 0) * 2  # 减去阈值

   # 将小于0的重置为0
   img = np.maximum(img, 0)

   # 中值滤镜更好地保留边缘
   img = ndimage.median_filter(img, 5)

   # 如果原始影像像素值为0，即没有数据不做检测，将img对应的位置像素值设为0
   mask1 = (R1 != 0).astype(np.float32)
   mask2 = (R2 != 0).astype(np.float32)
   img = img * mask1 * mask2

   return img,geoTrans1

def Image_contrast_detection2(file1, file2, img_save):
   '''
   2个文件夹中的图像对比检测，返回最后的输出所有图像合并结果图(按文件夹处理)
   :param file1_path: 文件夹1
   :param file2_path: 文件夹2
   :param img_save: 输出结果保存路径
   :return: 文件夹1,2对应图像对比结果
   '''

   # 为了支持中文路径，请添加下面这句代码
   gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")

   gdal.AllRegister()  # 注册驱动

   # assert len(sys.argv) >= 4, "参数不足 使用参考:\n python test.py 图一路径 图二路径 输出结果路径"

   # img1_path = sys.argv[1]
   # img2_path = sys.argv[2]

   [img1_paths,img_paths]=get_tif_path(file1,file2)

   info_struct=get_geoPosition(img1_paths)
   img1_paths=None
   flag = True
   i=0
   # data_type = np.float32
   for img_path in img_paths:

       sys.stdout.write('\r>> The program is calculating... %d/%d' % (
         i + 1, len(img_paths)))  # 输出进度条
       sys.stdout.flush()

       # img_name = img_path[0].split('/')[-1].split('.')[0]  # Linux 为'/' ;windows 为'\\'
       img,geoTrans=Image_detection(img_path)

       if flag:
          # 输出结果图像

          # raster_fn = path.join(sys.argv[3], img_name + '_mask.tif')

          raster_fn = img_save+'_mask.tif'

          # target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, xcount1, ycount1, 1, gdal.GDT_Byte)
          target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, int(info_struct[2]), int(info_struct[3]), 1, gdal.GDT_Byte)
          target_ds.SetGeoTransform(info_struct[0])  # 设置掩膜的地理参考
          target_ds.SetProjection(info_struct[1])  # 设置掩膜坐标引用
          flag=False

       # 反算出在合并图中的行列坐标
       xoff = np.ceil((geoTrans[0] - info_struct[0][0]) / geoTrans[1])
       yoff = np.ceil((geoTrans[3] - info_struct[0][3]) / geoTrans[5])

       target_ds.GetRasterBand(1).WriteArray(img, int(xoff), int(yoff))
       # target_ds.GetRasterBand(1).WriteRaster(0,0,xcount1,ycount1,img.tobytes())
       # [target_ds.GetRasterBand(1).WriteArray(img, 0, i) for i in ycount1]
       target_ds.FlushCache()
       i += 1
   sys.stdout.write('\n')
   sys.stdout.flush()
   target_ds = None


if __name__=="__main__":
   assert len(sys.argv) >= 4, "参数不足 使用参考:\n python test.py 图一文件夹 图二文件夹 输出结果路径"

   file1_path = sys.argv[1]
   file2_path = sys.argv[2]
   img_save=sys.argv[3]

   # file1_path = r'E:\05'
   # file2_path = r"E:\06"
   # img_save = r'E:\\'

   start_time = datetime.now()
   Image_contrast_detection2(file1_path, file2_path, img_save)
   end_time = datetime.now()
   print((end_time - start_time).total_seconds())
