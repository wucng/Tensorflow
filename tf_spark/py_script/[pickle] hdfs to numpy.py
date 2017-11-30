#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
mnist 转成 hdfs  参考：https://github.com/fengzhongyouxia/Tensorflow/blob/master/tf_spark/README.md
hdfs 转 numpy 参考：http://blog.csdn.net/wc781708249/article/details/78251701#t3
tensorflow 读取hdfs https://github.com/fengzhongyouxia/Tensorflow/blob/master/tf_Distributed/Read_HDFS.md

执行命令：spark-submit test.py

或

spark-submit \
--queue default \
--num-executors 45 \
--executor-memory 2G \
--driver-memory 12G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
test.py

或

spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 45 \
--executor-memory 2G \
--driver-memory 12G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
test.py

"""


from pyspark import SparkContext,SparkConf
import numpy as np
# import pickle

index=["part-00000","part-00001","part-00002","part-00003","part-00004","part-00005","part-00006",
       "part-00007","part-00008","part-00009"]
mode=["train","test"]
img_label=["images","labels"]

dirPath='hdfs://xxx:8020/user/root/mnist/pickle/' # 注该数据为pickle格式

# 解析路径
train_img_paths=[]
[train_img_paths.append(dirPath+mode[0]+'/'+img_label[0]+'/'+i) for i in index]
train_label_paths=[]
[train_label_paths.append(dirPath+mode[0]+'/'+img_label[1]+'/'+i) for i in index]

test_img_paths=[]
[test_img_paths.append(dirPath+mode[1]+'/'+img_label[0]+'/'+i) for i in index]
test_label_paths=[]
[test_label_paths.append(dirPath+mode[1]+'/'+img_label[1]+'/'+i) for i in index]

sc = SparkContext(conf=SparkConf().setAppName("The first example"))
# textFiles=sc.textFile(dirPath) # 读取 txt，csv 格式数据 也可以读取本地数据
def get_data(paths):
    for i,train_img_path in enumerate(paths):
        textFiles=sc.pickleFile(train_img_path) # 读取pickle数据 也可以读取本地数据
        data=textFiles.collect()
        if i==0:
            data1 = np.array(data, np.float32)  # 转成array
        else:
            data1=np.vstack((data1,np.array(data, np.float32)))
    return data1

train_x=get_data(train_img_paths)
train_y=get_data(train_label_paths)

test_x=get_data(test_img_paths)
test_y=get_data(test_label_paths)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

'''
(60000, 784)
(60000, 10)
(10000, 784)
(10000, 10)
'''
