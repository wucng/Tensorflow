# -*- coding:utf-8 -*-
# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

"""
注：不要减小原来数据格式
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.conf import SparkConf

import argparse
import os
import numpy
import sys
import tensorflow as tf
import threading
import time
from datetime import datetime

# from com.yahoo.ml.tf import TFCluster
from tensorflowonspark import TFCluster
import mnist_dist

sc = SparkContext(conf=SparkConf().setAppName("mnist_spark")) # mnist_spark 可以自行修改
executors = sc._conf.get("spark.executor.instances") # spark worker实例个数
num_executors = int(executors) if executors is not None else 1
num_ps = 1 # ps（主）节点个数

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", help="number of records per batch", type=int, default=200) # 每步训练的样本数
parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=1) # 样本迭代总数
parser.add_argument("-f", "--format", help="example format: (csv|pickle|tfr)", choices=["csv","pickle","tfr"], default="csv") # 输入样本数据格式
parser.add_argument("-i", "--images", help="HDFS path to MNIST images in parallelized format") # HDFS 图像路径
parser.add_argument("-l", "--labels", help="HDFS path to MNIST labels in parallelized format") # HDFS 标签路径
parser.add_argument("-m", "--model", help="HDFS path to save/load model during train/inference", default="mnist_model") #模型保存位置
parser.add_argument("-n", "--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors) #集群规模大小
parser.add_argument("-o", "--output", help="HDFS path to save test/inference output", default="predictions") # 输出结果保存位置
parser.add_argument("-r", "--readers", help="number of reader/enqueue threads", type=int, default=1) # 读/队列的线程数量
parser.add_argument("-s", "--steps", help="maximum number of steps", type=int, default=1000) # 总的训练步数
parser.add_argument("-tb", "--tensorboard", help="launch tensorboard process", action="store_true") # 开启tensorboard
parser.add_argument("-X", "--mode", help="train|inference", default="train") # 模式 train表示 训练；inference 表示 推理
parser.add_argument("-c", "--rdma", help="use rdma connection", default=False) # RDMA 模式 远程直接数据存取

parser.add_argument("-md", "--model_name", help="The model name",type=str,default="model.ckpt")
# parser.add_argument("-md2", "--model_name2", help="The model name", type=str,default="model2.ckpt")
parser.add_argument("-a", "--acc", help="Precision threshold", type=float,default=0.5)
parser.add_argument("-dr", "--dropout", help="Retention rate", type=float,default=0.5)
parser.add_argument("-lr", "--learning_rate", help="learning rate", type=float,default=1e-6)
args = parser.parse_args()
print("args:",args)

print("{0} ===== Start".format(datetime.now().isoformat()))

if args.format == "tfr":  # HDFS==>numpy array
  images = sc.newAPIHadoopFile(args.images, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                              keyClass="org.apache.hadoop.io.BytesWritable",
                              valueClass="org.apache.hadoop.io.NullWritable")
  def toNumpy(bytestr):
    example = tf.train.Example()
    example.ParseFromString(bytestr)
    features = example.features.feature
    image = numpy.array(features['image'].int64_list.value)
    label = numpy.array(features['label'].int64_list.value)
    return (image, label)
  dataRDD = images.map(lambda x: toNumpy(str(x[0])))
else:
  if args.format == "csv": # HDFS==>numpy array
    images = sc.textFile(args.images).map(lambda ln: [int(x) for x in ln.split(',')])
    labels = sc.textFile(args.labels).map(lambda ln: [float(x) for x in ln.split(',')])
  else: # args.format == "pickle":  # HDFS==>numpy array
    images = sc.pickleFile(args.images)
    labels = sc.pickleFile(args.labels)

  print("zipping images and labels")
  # print(type(labels))
  # print(labels.count())
  dataRDD = images.zip(labels) # image+label

#cluster = TFCluster.reserve(sc, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
#cluster.start(mnist_dist.map_fun, args)
cluster = TFCluster.run(sc, mnist_dist.map_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
if args.mode == "train" or args.mode == "retrain":
  cluster.train(dataRDD, args.epochs)
else:
  labelRDD = cluster.inference(dataRDD)
  labelRDD.saveAsTextFile(args.output)
cluster.shutdown()  # 集群关闭

print("{0} ===== Stop".format(datetime.now().isoformat()))


