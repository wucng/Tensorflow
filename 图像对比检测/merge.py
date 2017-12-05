#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
图像变化检测的结果图合并成一张大图
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


def get_tif_path(file1):
   '''
   返回tif/tiff对应的路径
   :param file1: 文件夹
   :return: 返回做完匹配后结果图像的路径
   '''
   file1_path = []
   [file1_path.append(glob(os.path.join(file1, '*/' * i, '*.tif'))) for i in range(3)] # 只循环3级目录
   [file1_path.append(glob(os.path.join(file1, '*/' * i, '*.tiff'))) for i in range(3)]
   file1_path = list(itertools.chain.from_iterable(file1_path))  # 转成一维list

   return file1_path

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

def merge(file,img_save):
    '''
    将图像对比检测的结果图合并到一张图
    :param file: 图像检测的结果图文件夹
    :param img_save: 合并后的图像保存位置
    :return: 返回合并后的图
    '''
    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")

    gdal.AllRegister()  # 注册驱动
    img_paths=get_tif_path(file)

    info_struct = get_geoPosition(img_paths)

    data_type = np.float32
    flag = True
    i = 0
    for img_path in img_paths:
        sys.stdout.write('\r>> The program is calculating... %d/%d' % (
            i + 1, len(img_paths)))  # 输出进度条
        sys.stdout.flush()

        src_ds=gdal.Open(img_path, GA_ReadOnly)
        geoTrans = src_ds.GetGeoTransform()
        xcount = src_ds.RasterXSize
        ycount = src_ds.RasterYSize
        srcband = src_ds.GetRasterBand(1)
        img = srcband.ReadAsArray(0, 0, xcount, ycount).astype(data_type)

        if flag:
            # 输出结果图像

            # raster_fn = path.join(sys.argv[3], img_name + '_mask.tif')

            raster_fn = img_save + '_mask.tif'
            papszCreateOptions = ['compress=lzw'] # 压缩
            # target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, xcount1, ycount1, 1, gdal.GDT_Byte)
            target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, int(info_struct[2]), int(info_struct[3]), 1,
                                                             gdal.GDT_Byte,papszCreateOptions)
            target_ds.SetGeoTransform(info_struct[0])  # 设置掩膜的地理参考
            target_ds.SetProjection(info_struct[1])  # 设置掩膜坐标引用
            flag = False

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
   assert len(sys.argv) >= 3, "参数不足 使用参考:\n python test.py 文件夹 输出结果路径"

   file_path = sys.argv[1]
   img_save=sys.argv[2]

   # file1_path = r'E:\05'
   # file2_path = r"E:\06"
   # img_save = r'E:\\'

   start_time = datetime.now()
   merge(file_path, img_save)
   end_time = datetime.now()
   print((end_time - start_time).total_seconds())



