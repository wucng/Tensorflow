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

'''
数据转换的脚本，融合了 m1.py,tool_set.py，可以针对standalone和 yarn版本
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

"""
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
    # with gzip.open(dir_name + '/train_data.pkl', 'wb') as writer: # 以压缩包方式创建文件，进一步压缩文件
    #     pickle.dump(data, writer)  # 数据存储成pickle文件

"""

# -----------------------------------------------
'''
# 读取pickle文件
# filename 为生成的pickle 路径名
# img_pixel 为图像像素尺寸
# channels 为波段数
# return： 特征+标签
'''
def read_and_decode1(filename,img_pixel=60,channels=4):
    '''
    # 使用gzip  （如果生成pickle时使用gzip，这里对应必须使用）
    with gzip.open(filename, 'rb') as pkl_file: # 打开文件
        data = pickle.load(pkl_file) # 加载数据
    '''
    # 不使用gzip
    with open(filename, 'rb') as pkl_file: # 打开文件
        data = pickle.load(pkl_file) # 加载数据

    for i in range((data.shape)[0]):
        data_1 = np.fromstring(data[i, :])# 转码 先解压缩，再转成数组
        if i == 0:
            data_2 = data_1
        else:
            data_2 = np.vstack((data_2, data_1)) # 上下合并

    # data= data_2.reshape((-1, img_pixel * img_pixel * channels + 1))  # 数据每一行都是 图像+标签

    # 将数据按行打乱 这里不打乱 否则测试时 数据与图像名对应不起来
    # index = [i for i in range(len(data_2))]  # len(data_2)得到的行数
    # np.random.shuffle(index)  # 将索引打乱
    # data_2 = data_2[index]

    return data_2

# ------------------------------------------------

'''
# 读取.gz文件
# filename 为生成的pickle 路径名
# img_pixel 为图像像素尺寸
# channels 为波段数
# return： 特征+标签
'''
def read_and_decode2(filename,img_pixel=60,channels=4):
    flag=False
    # while(True):
    with gzip.open(filename, 'rb') as pkl_file:  # 打开文件
        for row in pkl_file:
            # row=pkl_file.readline()
            if not row or row == b'\n':
                break

            # data=row.replace('\n', '')  # 删除末尾的换行符
            data=row.replace(b'\n',b'') # 删除末尾换行符  因为row为byte类型 前面要加b 不加为字符串
            # data_1 = np.fromstring(zlib.decompress(data))  # 转码 先解压缩，再转成数组

            data_1 = np.fromstring(data)
            if flag==False:
                data_2 = data_1
                flag=True
            else:
                data_2 = np.vstack((data_2, data_1))  # 上下合并

    return data_2

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
        # 如果labels_dense不是int类型，itenm就不是int，此时做数组的切片索引就会报错，
        # 数组索引值必须是int类型，也可以 int(itenm) 强制转成int
        # labels_one_hot[i, :][itenm] = 1
    return labels_one_hot

# -----------------------------------------------

# def writeMNIST(sc, input_images, input_labels, output, format, num_partitions)
def writeMNIST(sc, dir_name,img_pixel,channels, output, format, num_partitions): # 按照自己数据格式相应修改
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
  # load MNIST gzip into memory
  with open(input_images, 'rb') as f:
    images = numpy.array(mnist.extract_images(f)) # 将原有的图像数据全部提取出来转成numpy array

  with open(input_labels, 'rb') as f: # 将原有的标签数据全部提取出来转成numpy array
    if format == "csv2": # 数据格式
      labels = numpy.array(mnist.extract_labels(f, one_hot=False)) # array
    else:
      labels = numpy.array(mnist.extract_labels(f, one_hot=True)) # array

  '''

  # tool_set.create_pickle_train(dir_name,img_pixel,channels)

  # data=tool_set.read_and_decode(dir_name+"/train_data.pkl",img_pixel,channels)

  '''
  # 直接读取图像
  data=create_pickle_train(dir_name,img_pixel,channels)
  '''

  '''
  # 读取pickle
  data=read_and_decode1(dir_name,img_pixel,channels)
  '''

  # 读取.gz文件
  # data = read_and_decode2(dir_name, img_pixel, channels)
  # '''
  data=read_and_decode(dir_name)

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

  #标签--->float，图像数据-->int
  #下面这个处理一定要加，否则后续训练由于数据类型与TensorFlowOnSpark自带的数据类型
  # 不一致，而产生各种错误
  labels=labels.astype(np.float16)
  images=images.astype(np.uint8)

  # 如果使用自己的数据转成HDFS 需要修改 上面两个 open 将自己的数据 转成 numpy array

  shape = images.shape # 图像总数 x 28 x 28 x 1(波段数)
  print("images.shape: {0}".format(shape))          # 60000 x 28 x 28  mnist数据 28x28x1 0~9(10类)
  print("labels.shape: {0}".format(labels.shape))   # 60000 x 10

  # create RDDs of vectors
  imageRDD = sc.parallelize(images.reshape(shape[0], shape[1] * shape[2]*shape[3]), num_partitions) # [-1,28*28*1]
  # imageRDD = sc.parallelize(images.reshape(shape[0], shape[1] * shape[2]*nBands), num_partitions)  nBands 图像波段数

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
    '''
    针对standalone版本
    直接读取图像/pickle文件/.gz文件
    writeMNIST(sc, args.dirTrain, args.imgPixel,args.channels, args.output + "/train", args.format, args.num_partitions) # 转换train数据
    writeMNIST(sc, args.dirTest, args.imgPixel,args.channels, args.output + "/test", args.format, args.num_partitions) # 转换test数据
    '''
    # 针对yarn版本
    # 读取.gz/pickle文件 必须先将train_data.txt.gz与test_data.txt.gz压缩成mnist.zip  再将mnist.zip上传到 hdfs上
    writeMNIST(sc, 'mnist/11.pkl', args.imgPixel, args.channels, args.output + "/train", args.format,
               args.num_partitions)  # 转换train数据
    writeMNIST(sc, 'mnist/11.pkl', args.imgPixel, args.channels, args.output + "/test", args.format,
               args.num_partitions)  # 转换test数据

  if args.read or args.verify: # 读取 or 验证 样本
    readMNIST(sc, args.output + "/train", args.format)


# 如果是自己的数据需要转成 HDFS 需要修改的地方
# num-partitions 样本分类数
# writeMNIST 中的 两个Open，只需实现将自己的数据转成numpy即可
# writeMNIST 中的 imageRDD 加上图像波段数

# standalone版本 提交命名
'''
${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
test2/mnist_data_setup.py \
--output test2/csv \
--format csv \
--dirTrain test2/train_data.pkl \
--dirTest test2/test_data.pkl \
--imgPixel 10 \
--channels 4 \
--num-partitions 2
'''

# yarn版本 提交命名
'''
zip -r mnist.zip train_data.txt.gz test_data.txt.gz  # 将train_data.txt.gz与test_data.txt.gz压缩成mnist.zip
hdfs dfs -rm test2/mnist.zip  # 先删除原有的文件(如果文件存在)
hdfs dfs -put mnist.zip test2 # 将mnist.zip上传到 hdfs上  hdfs dfs -ls test2 查看

spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 4 \
--executor-memory 27G \
--archives hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/test2/mnist.zip#mnist \ # 这个mnist.zip就包含了训练和测试文件 不需要重新传入训练和测试文件的路径（这点不同于standalone版本）
mnist_data_setup.py \
--output hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/test2/csv \
--imgPixel 10 \
--channels 3 \
--num-partitions 2
'''

"""
spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 4 \
--executor-memory 12G \
--driver-memory 4G \   # 如果出现内存不足，加上这条
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/test2/mnist.zip#mnist \
mnist_data_setup.py \
--output hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/test2/csv \
--imgPixel 10 \
--channels 3 \
--num-partitions 2

"""

