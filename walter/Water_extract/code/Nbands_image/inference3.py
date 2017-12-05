#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
传入一张影像进行预测，并提取水体范围
"""

from __future__ import print_function
# import os
# from os import path
import tensorflow as tf
from PIL import Image
import numpy as np
# import tool_set
# import pandas as pd
# import matplotlib.pyplot as plt
# import pickle
import datetime
import numexpr as ne #


# starttime = datetime.datetime.now()
print('start time:',datetime.datetime.now())
# 训练集文件路径

# dir_name=r'F:\PL\2\1234511.tif' # 影像路径
dir_name="../../../Water_extract/data/1234511.tif"

'''
# CNN 完整程序  测试模型
'''
# Parameters
# learning_rate = 10**(-5)
training_epochs = 500
# training_iters = 200000
batch_size = 128
display_step = 1
img_pixel=10
channels=4

# Network Parameters
img_size = img_pixel*img_pixel*channels # data input (img shape: 28*28*3)
label_cols = 2 # total classes (云、非云)使用标签[1,0,0] 3维
dropout = 0.75 # Dropout, probability to keep units
img_nums=6000


# ---------设置动态学习效率
# Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = batch_size      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

global_step=training_epochs*(img_nums//batch_size)   # Integer Variable counting the number of training steps
# Variables that affect learning rate.
num_batches_per_epoch = img_nums / batch_size
decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY) # 多少步后开始执行学习效率衰减

# Decay the learning rate exponentially based on the number of steps.
learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                global_step,
                                decay_steps,
                                LEARNING_RATE_DECAY_FACTOR,
                                staircase=True)
# 设置动态学习效率----------


with tf.device('/gpu:0'):
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, img_size]) # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, label_cols]) #
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)   # strides中间两个为1 表示x,y方向都不间隔取样
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME') # strides中间两个为2 表示x,y方向都间隔1个取样

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, img_pixel, img_pixel, channels]) # 彩色图像3个频道，如果是灰度图就是1

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1']) # 图像60*60*32
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2) # 图像 14*14*32

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2']) # 图像 30*30*64
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2) # 图像 15*15*64

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]]) # [None,15*15*64]
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    # out=tf.nn.softmax(out)  # softmax得到最后预测标签
    # out=tf.nn.sigmoid(out)
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 3 input, 32 outputs 彩色图像3个输入(3个频道)，灰度图像1个输入
    'wc1': tf.Variable(tf.random_normal([5, 5, channels, 32])), # 5X5的卷积模板
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([(1+img_pixel//4)*(1+img_pixel//4)*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, label_cols]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([label_cols]))
}


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("Create Graph:",datetime.datetime.now())

#初始化所有的op
# init = tf.global_variables_initializer()

from osgeo import gdal, ogr
from osgeo.gdalconst import *

isize=10 # 10 x 10的样本
m=1
isizes=m*isize

# dBuf_0=np.zeros((isizes,isizes),np.uint8) # 作为输出掩膜的像素
if __name__ == '__main__':
    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")

    gdal.AllRegister()  # 注册驱动
    ogr.RegisterAll()

    srcDS = gdal.Open(dir_name, GA_ReadOnly)  # 只读方式打开原始影像

    geoTrans = srcDS.GetGeoTransform()  # 获取地理参考6参数
    srcPro = srcDS.GetProjection()  # 获取坐标引用
    srcXSize = srcDS.RasterXSize  # 宽度
    srcYSize = srcDS.RasterYSize  # 高度
    nbands = srcDS.RasterCount  # 波段数

    # raster_fn = path.join(dir_name, 'test_mask.tiff')  # 存放掩膜影像
    raster_fn=dir_name+"test_mask_0.tiff"
    # raster_fn ="test_mask_1.tiff"

    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, srcXSize, srcYSize, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geoTrans)  # 设置掩膜的地理参考
    target_ds.SetProjection(srcPro)  # 设置掩膜坐标引用
    # band = target_ds.GetRasterBand(1)  # 获取第一个波段(影像只有一个波段)
    # band.SetNoDataValue(0)  # 将这个波段的值全设置为0
    target_ds.GetRasterBand(1).SetNoDataValue(0)

    del geoTrans
    del srcPro
    del dir_name,raster_fn

    dBuf_1=np.zeros((srcXSize,srcYSize),np.uint8) # 整个掩膜的缓存

    saver = tf.train.Saver()  # 默认是保存所有变量
    with tf.Session() as sess:
        print('session start:',datetime.datetime.now())
        saver.restore(sess, "./outfile/model/save_net.ckpt")  # 提取模型数据
        flagY = True
        ne.set_num_threads(100) #设置4个线程
        for i in range(0, srcYSize,isizes):
            # i=int(i)
            print("正在计算：%.2f%%" % (i/srcYSize * 100), end="\r")
            if not flagY: break
            if i + isizes > srcYSize - 1:
                i = srcYSize - 1 - isizes
                flagY = False

            flagX = True
            for j in range(0,srcXSize,isizes):
                # j = int(j)
                if not flagX: break

                if j + isizes> srcXSize - 1:
                    j = srcXSize - 1 - isizes
                    flagX = False

                for band in range(nbands):
                    # band=int(band)
                    band += 1
                    # print "[ GETTING BAND ]: ", band
                    srcband = srcDS.GetRasterBand(band)  # 获取该波段
                    if srcband is None:
                        continue

                    # Read raster as arrays 类似RasterIO（C++）
                    dataraster = srcband.ReadAsArray(j, i, isizes, isizes,isizes, isizes).astype(np.uint8) # 像素值转到[0~255]

                    if band == 1:
                        data = dataraster.reshape((isizes, isizes, 1))
                    else:
                        # 将每个波段的数组很并到一个3s维数组中
                        data = np.append(data, dataraster.reshape((isizes, isizes, 1)), axis=2)
                    del dataraster,srcband


                # print('loop start:', datetime.datetime.now())

                data_2_1=data.reshape([-1, img_size])
                data_2_1 = data_2_1 * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
                pred2 = sess.run(pred, feed_dict={x: data_2_1, keep_prob: 1.})
                del data_2_1
                if sess.run(tf.argmax(pred2, 1))[0]:

                    tt=1
                    print('loop2 start:', datetime.datetime.now())
                    for xx in range(isize):
                        # xx=int(xx)
                        # print('loop2 start:', datetime.datetime.now())
                        for y in range(isize):
                            # y=int(y)
                            image3 = Image.fromarray(np.uint8(data[xx:xx+1, y:y+1, :]))

                            data_3 = image3.resize([isize, isize])  # (isize x isize x 4)
                            del image3
                            # data_3=list(data[xx:xx+1, y:y+1, :])*(isize*isize)
                            data_3 = np.array(data_3,np.uint8)  # 转成array
                            # data_3=data_3.reshape([isize,isize,4])

                            # 数据归一化
                            data_3 = data_3 * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
                            data_3 = data_3.reshape([-1, img_size])
                            pred3 = sess.run(pred, feed_dict={x: data_3, keep_prob: 1.})

                            del data_3
                            if sess.run(tf.argmax(pred3, 1))[0]:
                                dBuf_1[i:i+isize,j:j+isize][tt*xx:tt*xx+tt, tt*y:tt*y+tt] = np.ones((tt,tt),np.uint8)*255
                            del pred3

                del pred2


        print('writer data:', datetime.datetime.now())
        target_ds.GetRasterBand(1).WriteArray(dBuf_1, 0, 0) # 数据写入要比数据计算慢，会导致程序卡主，所以减少写入次数，提升程序运行速度
        target_ds.FlushCache()  # 将数据写入文件
        target_ds=None
        print("计算完成：100.00%", end="\r")
        # endtime = datetime.datetime.now()
        print('end time:',datetime.datetime.now())
        # print((endtime - starttime).seconds)
        exit()
