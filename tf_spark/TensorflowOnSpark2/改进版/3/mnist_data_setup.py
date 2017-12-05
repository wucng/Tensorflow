# -*- coding:utf-8 -*-
# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from array import array
import os
# from tensorflow.contrib.learn.python.learn.datasets import mnist

import sys
import pickle
import gzip
import os

'''
数据转换的脚本，融合了 m1.py,tool_set.py，可以针对standalone和 yarn版本
实现从gt-data 读取影像数据,也可以从hdfs上读取数据
'''

def toTFExample(image, label):  # array--->tfrecode
  '''
  :param image:  numpy array
  :param label:  numpy array
  :return: TFExample字节字符串
  '''
  """Serializes an image/label as a TFExample byte string"""
  '''序列化一个图像/标签作为TFExample字节字符串'''
  example = tf.train.Example(
    features = tf.train.Features(
      feature = {
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label.astype("int64"))),
        'image': tf.train.Feature(int64_list=tf.train.Int64List(value=image.astype("int64")))
      }
    )
  )
  return example.SerializeToString()

def fromTFExample(bytestr): #tfrecode-->array
  """Deserializes a TFExample from a byte string"""
  example = tf.train.Example()
  example.ParseFromString(bytestr)
  return example

def toCSV(vec): # array--> csv数据
  """Converts a vector/array into a CSV string"""
  return ','.join([str(i) for i in vec])

def fromCSV(s):# 从csv-->array
  """Converts a CSV string to a vector/array"""
  return [float(x) for x in s.split(',') if len(s) > 0]


try:
  from osgeo import gdal
except:
  import gdal

#-------------------------------------------------
'''
# 多波段图像(遥感图像)提取每个波段信息转换成数组（波段数>=4 或者 波段数<=3）
# 一般的方法如：opencv，PIL，skimage 最多只能读取3个波段
# path 图像的路径
# return： 图像数组
'''
def Multiband2Array(path):

    src_ds = gdal.Open(path)
    if src_ds is None:
        print('Unable to open %s'% path)
        sys.exit(1)

    xcount=src_ds.RasterXSize # 宽度
    ycount=src_ds.RasterYSize # 高度
    ibands=src_ds.RasterCount # 波段数

    # print "[ RASTER BAND COUNT ]: ", ibands
    for band in range(ibands):
        band += 1
        # print "[ GETTING BAND ]: ", band
        srcband = src_ds.GetRasterBand(band) # 获取该波段
        if srcband is None:
            continue

        # Read raster as arrays 类似RasterIO（C++）
        dataraster = srcband.ReadAsArray(0, 0, xcount, ycount).astype(np.float32)
        if band==1:
            data=dataraster.reshape((xcount,ycount,1))
        else:
            # 将每个波段的数组很并到一个3维数组中
            data=np.append(data,dataraster.reshape((xcount,ycount,1)),axis=2)

    return data

# -----------------------------------------------
'''
# dir_name 为文件路径
# img_pixel 为图像的像素尺寸
'''
def create_pickle_train(dir_name,img_pixel=60,channels=4):
    flag=False
    for _,dirs,_ in os.walk(dir_name):
        for filename in dirs: # 文件夹名 取文件名作为标签
            file_path=os.path.join(dir_name,filename) # 文件夹路径
            # for _ , _,img in os.walk(file_path):
            for img_name in os.listdir(file_path): # 文件夹下的文件名
                imgae_path = os.path.join(file_path, img_name) # 文件路径

                img=Multiband2Array(imgae_path) # 使用GDAL方法读取影像 可以读取多于3个波段的影像

                data1=img.reshape((-1,img_pixel*img_pixel*channels)) # 展成一行
                label=np.array([int(filename)]) # 文件名作为标签

                data2=np.append(data1,label)[np.newaxis,:] # 数据+标签

                # data2=data2.tostring()  # 转成byte，缩小文件大小
                # data2=zlib.compress(data2) # 使用zlib将数据进一步压缩


                if flag==False:
                    data=data2
                if flag==True:
                    data=np.vstack((data,data2))  # 上下合并
                flag = True
    return data


# -------------------------------------------------

def create_pickle_train(image_path, mask_path, img_pixel=10, channels=3):
    # m = 0
    # image_data = Multiband2Array(image_path)
    image_data=Multiband2Array(image_path)
    # mask_data = cv2.split(cv2.imread(mask_path))[0] / 255
    # mask_data=np.asarray(Image.open(mask_path))//255

    mask_data=Multiband2Array(mask_path)//255


    x_size, y_size = image_data.shape[:2]

    data_list = []
    for i in range(0, x_size - img_pixel + 1, img_pixel // 2):  # 文件夹下的文件名
        if i + img_pixel > x_size:
            i = x_size - img_pixel - 1
        for j in range(0, y_size - img_pixel + 1, img_pixel // 2):
            if j + img_pixel > y_size:
                j = y_size - img_pixel - 1
            cropped_data = image_data[i:i + img_pixel, j:j + img_pixel]
            data1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))  # 展成一行
            train_label = mask_data[i:i + img_pixel, j:j + img_pixel].max()
            # train_label = 1
            # train_label = mask_data[i:i + img_pixel, j:j + img_pixel].min()
            # train_label = int(mask_data[i:i + img_pixel, j:j + img_pixel].sum() / (img_pixel*img_pixel/2+1))
            data2 = np.append(data1, train_label)[np.newaxis, :]  # 数据+标签
            data_list.append(data2)

            # if m % 10000 == 0: print(datetime.datetime.now(), "compressed {number} images".format(number=m))

    data_matrix = np.array(data_list, dtype=np.float32)
    data_matrix = data_matrix.reshape((-1, 301))
    return data_matrix


# -------------------------------------------------
def read_and_decode(filename):
    with gzip.open(filename, 'rb') as pkl_file:  # 打开文件
        data = pickle.load(pkl_file)  # 加载数据

    return data
# -------------------------------------------------
def dense_to_one_hot2(labels_dense,num_classes):
    labels_dense=np.array(labels_dense,dtype=np.uint8)
    num_labels = labels_dense.shape[0] # 标签个数
    labels_one_hot=np.zeros((num_labels,num_classes),np.uint8)
    for i,itenm in enumerate(labels_dense):
        labels_one_hot[i,itenm]=1
    return labels_one_hot

# -----------------------------------------------

# def writeMNIST(sc, input_images, input_labels, output, format, num_partitions)
def writeMNIST(sc, dir_name,dir_name1,img_pixel,channels, output, format, num_partitions): # 按照自己数据格式相应修改
  '''
  :remark  将原有的数据转成需要的格式 存储成HDFS
  :param sc:  SparkContext 不改
  :param dir_name:  输入存放影像路径 如：存有0,1文件夹（2类）
  :param img_pixel:   图像大小（如：mnist 28）
  :param channels:    图像波段数（如：mnist 波段为 1）
  :param output:    转成HDFS输出路径
  :param format:    需要转成的数据格式
  :param num_partitions:  实际图像分类数  mnist 10分类 所有为10
  :return: HDFS
  '''
  """Writes MNIST image/label vectors into parallelized files on HDFS"""
  '''
  '''
  # data=read_and_decode(dir_name)

  data=create_pickle_train(dir_name,dir_name1,img_pixel,channels)

  ## 图像-->numpy array
  # data = create_pickle_train(dir_name,img_pixel,channels) #(image+label)

  # 将数据按行打乱
  index = [i for i in range(len(data))]  # len(data)得到的行数
  np.random.shuffle(index)  # 将索引打乱
  data = data[index]
  del index


  labels_dense=data[:,-1] #取出标签列
  if format == "csv2":  # 数据格式
    labels=labels_dense
  else:
    # 转成one_hot
    labels=dense_to_one_hot2(labels_dense,num_partitions)
  del labels_dense

  images_=data[:, 0:img_pixel * img_pixel * channels]
  images=images_.reshape((-1,img_pixel,img_pixel,channels))
  del data

  labels=labels.astype(np.float16)
  images=images.astype(np.float16)

  shape = images.shape # 图像总数 x 28 x 28 x 1(波段数)
  print("images.shape: {0}".format(shape))          # 60000 x 28 x 28  mnist数据 28x28x1 0~9(10类)
  print("labels.shape: {0}".format(labels.shape))   # 60000 x 10

  # create RDDs of vectors
  imageRDD = sc.parallelize(images.reshape(shape[0], shape[1] * shape[2]*shape[3]), num_partitions) # [-1,28*28*1]

  labelRDD = sc.parallelize(labels, num_partitions)

  output_images = output + "/images" # 输出路径
  output_labels = output + "/labels" # 输出路径

  # save RDDs as specific format
  if format == "pickle":
    imageRDD.saveAsPickleFile(output_images) #保存成Pickle
    labelRDD.saveAsPickleFile(output_labels) #
  elif format == "csv":
    imageRDD.map(toCSV).saveAsTextFile(output_images) # 转成csv 再转成 Text
    labelRDD.map(toCSV).saveAsTextFile(output_labels) # 转成csv 再转成 Text
  elif format == "csv2":
    imageRDD.map(toCSV).zip(labelRDD).map(lambda x: str(x[1]) + "|" + x[0]).saveAsTextFile(output) # image + label 放在一个文件转成 text
  else: # format == "tfr":
    tfRDD = imageRDD.zip(labelRDD).map(lambda x: (bytearray(toTFExample(x[0], x[1])), None)) # 转成 .tfrecord
    # requires: --jars tensorflow-hadoop-1.0-SNAPSHOT.jar
    tfRDD.saveAsNewAPIHadoopFile(output, "org.tensorflow.hadoop.io.TFRecordFileOutputFormat",
                                keyClass="org.apache.hadoop.io.BytesWritable",
                                valueClass="org.apache.hadoop.io.NullWritable")
#  Note: this creates TFRecord files w/o requiring a custom Input/Output format
#  else: # format == "tfr":
#    def writeTFRecords(index, iter):
#      output_path = "{0}/part-{1:05d}".format(output, index)
#      writer = tf.python_io.TFRecordWriter(output_path)
#      for example in iter:
#        writer.write(example)
#      return [output_path]
#    tfRDD = imageRDD.zip(labelRDD).map(lambda x: toTFExample(x[0], x[1]))
#    tfRDD.mapPartitionsWithIndex(writeTFRecords).collect()

def readMNIST(sc, output, format): # 该函数不修改
  '''
  remark: 验证刚才转换的数据
  :param sc: SparkContext 不改
  :param output: 存放数据的位置
  :param format: 存放数据的格式
  :return: 返回数据验证信息
  '''
  """Reads/verifies previously created output"""

  output_images = output + "/images"
  output_labels = output + "/labels"
  imageRDD = None
  labelRDD = None

  if format == "pickle":
    imageRDD = sc.pickleFile(output_images)
    labelRDD = sc.pickleFile(output_labels)
  elif format == "csv":
    imageRDD = sc.textFile(output_images).map(fromCSV)
    labelRDD = sc.textFile(output_labels).map(fromCSV)
  else: # format.startswith("tf"):
    # requires: --jars tensorflow-hadoop-1.0-SNAPSHOT.jar
    tfRDD = sc.newAPIHadoopFile(output, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                              keyClass="org.apache.hadoop.io.BytesWritable",
                              valueClass="org.apache.hadoop.io.NullWritable")
    imageRDD = tfRDD.map(lambda x: fromTFExample(str(x[0])))

  num_images = imageRDD.count() # 影像数
  num_labels = labelRDD.count() if labelRDD is not None else num_images #标签与影像数对应 标签数
  samples = imageRDD.take(10) #取出10个
  print("num_images: ", num_images)
  print("num_labels: ", num_labels)
  print("samples: ", samples)# 打印10张影像的矩阵信息

if __name__ == "__main__":
  import argparse

  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf

  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--format", help="output format", choices=["csv","csv2","pickle","tf","tfr"], default="csv") #数据格式
  parser.add_argument("-d1","--dirTrain",help="The training image file  path",type=str,default="")
  parser.add_argument("-d2", "--dirTest", help="The test image file  path", type=str, default="")
  parser.add_argument("-i", "--imgPixel", help="The image pixel size", type=int,default=10)
  parser.add_argument("-c", "--channels", help="The channel of the image", type=int, default=3)
  parser.add_argument("-n", "--num-partitions", help="Number of output partitions", type=int, default=2) # 分类数  按实际分类数 修改
  parser.add_argument("-o", "--output", help="HDFS directory to save examples in parallelized format", default="mnist_data") #HDFS保存位置
  parser.add_argument("-r", "--read", help="read previously saved examples", action="store_true") #读取保存样本
  parser.add_argument("-v", "--verify", help="verify saved examples after writing", action="store_true") #验证样本

  args = parser.parse_args() # 参数变量
  print("args:",args) # 打印所有变量

  sc = SparkContext(conf=SparkConf().setAppName("mnist_parallelize"))# "mnist_parallelize" 名字 可以修改


  if not args.read: # 写入
    # Note: these files are inside the mnist.zip file
    writeMNIST(sc, 'gtdata:///users/xiaoshaolin/tensorflow/11.tif', 'gtdata:///users/xiaoshaolin/tensorflow/11_mask.tif',
              args.imgPixel, args.channels, args.output + "/train", args.format,
              args.num_partitions)  # 转换train数据  # 从gtdata上读取数据
    # writeMNIST(sc, 'mnist/11.tif', 'mnist/11_mask.tif',
    #            args.imgPixel, args.channels, args.output + "/train", args.format,
    #            args.num_partitions)  # 转换train数据
    # writeMNIST(sc, 'mnist/33.pkl', args.imgPixel, args.channels, args.output + "/train6", args.format,
    #            args.num_partitions)  # 转换train数据
    # writeMNIST(sc, 'mnist/34_0.pkl', args.imgPixel, args.channels, args.output + "/train7", args.format,
    #            args.num_partitions)  # 转换train数据
    #writeMNIST(sc, 'mnist/12.pkl', args.imgPixel, args.channels, args.output + "/test", args.format,
    #           args.num_partitions)  # 转换test数据

  if args.read or args.verify: # 读取 or 验证 样本
    readMNIST(sc, args.output + "/train", args.format)


"""
提交命令：
读取hdfs上的数据 先需把文件压缩成mnist.zip，再上传到hdfs上
spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 8 \
--executor-memory 32G \
--driver-memory 6G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/test2/mnist.zip#mnist \
--jars hdfs://dm01-08-01.tjidc.dcos.com:8020/spark-tensorflow/spark-tensorflow-connector-1.0-SNAPSHOT.jar \
mnist_data_setup.py \
--imgPixel 10 \
--channels 3 \
--output hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/test2/pickle \
--format pickle


读取gtdata 
spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 8 \
--executor-memory 32G \
--driver-memory 6G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--jars hdfs://dm01-08-01.tjidc.dcos.com:8020/spark-tensorflow/spark-tensorflow-connector-1.0-SNAPSHOT.jar \
mnist_data_setup.py \
--imgPixel 10 \
--channels 3 \
--output hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/pickle \
--format pickle
"""


