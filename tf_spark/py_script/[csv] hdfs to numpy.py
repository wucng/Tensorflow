#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
有问题，待解决
mnist 转成 hdfs  参考：https://github.com/fengzhongyouxia/Tensorflow/blob/master/tf_spark/README.md
hdfs 转 numpy 参考：http://blog.csdn.net/wc781708249/article/details/78251701#t3
tensorflow 读取hdfs https://github.com/fengzhongyouxia/Tensorflow/blob/master/tf_Distributed/Read_HDFS.md

执行命令：spark-submit test.py

"""

from pyspark import SparkContext,SparkConf
import numpy as np
# import pickle

mode=["train","test"]
img_label=["images","labels"]

dirPath='hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/mnist/csv/' # 注该数据为csv格式

# 解析路径
train_img_paths=[]
train_img_paths.append(dirPath+mode[0]+'/'+img_label[0])
train_label_paths=[]
train_label_paths.append(dirPath+mode[0]+'/'+img_label[1])

test_img_paths=[]
test_img_paths.append(dirPath+mode[1]+'/'+img_label[0])
test_label_paths=[]
test_label_paths.append(dirPath+mode[1]+'/'+img_label[1])

sc = SparkContext(conf=SparkConf().setAppName("The first example"))

def get_data(paths,cols):
   textFiles = sc.wholeTextFiles(paths) # 读取 txt、cvs文件
   data = textFiles.collect()
   for i in range(len(data)):
      data2 = data[i][1].encode("utf8").decode("utf8")
      if i==0:
         # data1 = np.reshape(np.fromstring(data2, np.float32),[-1,cols])
         data1 = np.fromstring(data2, np.float16)
         # print(data1.shape)
         print(data1[:10])
         exit(-1)
      else:
         # data1 = np.vstack((data1, np.reshape(np.fromstring(data2, np.float32),[-1,cols])))
         data1 = np.vstack((data1, np.fromstring(data2, np.float16)))

   return data1

train_x=get_data(train_img_paths[0],28*28*1)
train_y=get_data(train_label_paths[0],10)

test_x=get_data(test_img_paths[0],28*28*1)
test_y=get_data(test_label_paths[0],10)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
