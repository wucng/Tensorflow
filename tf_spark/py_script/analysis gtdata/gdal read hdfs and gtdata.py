#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
使用gdal 读取 hdfs上的图片，or 读取gtdata上的图片

执行方式：
Python xxx.py

or

spark-submit xxx.py

or

spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 30 \
--executor-memory 4G \
--driver-memory 12G \
# --py-files TensorFlowOnSpark/tfspark.zip,myfloder/inference/mnist_dist.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.yarn.executor.memoryOverhead=12288 \
xxx.py \
--parm1 \
--parm2 \
……
--parm \

"""


try:
  from osgeo import gdal
except:
  import gdal
import numpy as np
# from scipy import ndimage
import sys

def Multiband2Array(path):

    src_ds = gdal.Open(path)
    if src_ds is None:
        print('Unable to open %s'% path)
        sys.exit(1)

    xcount=src_ds.RasterXSize # 宽度
    ycount=src_ds.RasterYSize # 高度
    ibands=src_ds.RasterCount # 波段数

    # print "[ RASTER BAND COUNT ]: ", ibands
    # if ibands==4:ibands=3
    for band in range(ibands):
        band += 1
        # print "[ GETTING BAND ]: ", band
        srcband = src_ds.GetRasterBand(band) # 获取该波段
        if srcband is None:
            continue

        # Read raster as arrays 类似RasterIO（C++）
        dataraster = srcband.ReadAsArray(0, 0, xcount, ycount).astype(np.float16)
        if band==1:
            data=dataraster.reshape((ycount,xcount,1))
        else:
            # 将每个波段的数组很并到一个3维数组中
            data=np.append(data,dataraster.reshape((ycount,xcount,1)),axis=2)

    return data

# gdal 读取hdfs上的图片
path="HDFS://xxx:8020/user/root/mnist/0_0.bmp"
# path="HDFS://xxx:8020/user/root/pldata_03/20170821_092730_0c0b_3B_AnalyticMS.tif"

# hdfs 读取gtdata上的图片
# path="gtdata:///users/xxx/tensorflow/mnist_test/0/0_0.bmp"

data=Multiband2Array(path)
# data=ndimage.imread(path)

print(data.shape)
