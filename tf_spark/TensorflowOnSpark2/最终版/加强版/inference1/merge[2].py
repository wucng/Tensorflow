# -*- coding: utf-8 -*-
"""
将inference跑出的predictions，合并成一张完整的图像
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import glob
import argparse
import numpy as np
import re
import datetime
import os

from osgeo import gdal, ogr
from osgeo.gdalconst import *

"""
一次性合并所有影像
"""

parser = argparse.ArgumentParser()
parser.add_argument("-fp", "--file_path", help="file path",type=str,default="./predictions/") # 推理
parser.add_argument("-d", "--dir_name", help="Mask Image save the path",type=str,default="./images/") # 推理
args = parser.parse_args()
print("args:",args)

if not os.path.exists(args.dir_name):os.mkdir(args.dir_name)

file_name = []
[file_name.append(i.split('.')[0]) # 获取没有后缀的文件名
 for i in open('tile.log')] # 从hdfs上读取文件tile.log

for fileN in file_name:
    file_paths=glob.glob(args.file_path+fileN+'/'+'part*')
    if len(file_paths)==0:continue
    # print(file_paths)
    x_size=0
    y_size=0
    isize=2
    target_ds=None
    flag=True
    dBuf_1=None
    raster_fn = args.dir_name + fileN+"_mask.tif"

    start = datetime.datetime.now()
    print('session start:', datetime.datetime.now())

    for file_path in file_paths:
        reader=csv.reader(open(file_path))
        for item in reader:
            # print(type(item)) # list
            # print(type(item[1]))  # str
            # info=item[0][8:-1] # str
            info=re.split(r"\s+",item[0][8:-1]) # info=info.split(' ')
            # print(info)
            if flag:
                try:
                    y_size = int(float(info[-1][:-1]))
                    x_size=int(float(info[-2][:-1]))
                except:
                    y_size = int(float(info[-1]))
                    x_size = int(float(info[-2]))
                target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_size, y_size, 1, gdal.GDT_Byte)
                target_ds.GetRasterBand(1).SetNoDataValue(0)
                dBuf_1 = np.zeros([y_size, x_size], np.uint8)  # 整个掩膜的缓存
                flag=False
            try:
                j = int(float(info[-3][:-1]))
                i=int(float(info[-4][:-1]))
            except:
                j = int(float(info[-3]))
                i = int(float(info[-4]))
            # print('i',i,'j',j,'x_size',x_size,'y_size',y_size)
            pred=int(item[1][-1])
            if pred:
                dBuf_1[i:i + isize, j:j + isize] = np.ones([isize, isize], np.uint8) * 255

    print(datetime.datetime.now() - start)
    target_ds.GetRasterBand(1).WriteArray(dBuf_1, 0, 0)
    target_ds.FlushCache()  # 将数据写入文件
    target_ds = None
    dBuf_1=None
    print('end time:', datetime.datetime.now())
exit()
