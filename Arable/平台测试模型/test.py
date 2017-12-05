#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
from os import path
import tensorflow as tf
import numpy as np
import datetime
from osgeo import gdal, ogr
import tool_set
import cv2
import gzip
import pickle
import sys


def main(argv):
    starttime = datetime.datetime.now()
    # time格式化
    print("startTime: ", starttime)

    step = int(argv[3])
    dir_name = argv[2]  # 影像路径
    checkpoint_dir = argv[1]

    isize = 10
    img_pixel = isize
    channels = 3

    # Network Parameters
    img_size = isize * isize * channels  # data input (img shape: 28*28*3)
    label_cols = 2  # total classes (云、非云)使用标签[1,0,0] 3维

    with tf.device('/cpu:0'):
        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, img_size])  # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, label_cols])  #
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)  # strides中间两个为1 表示x,y方向都不间隔取样
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')  # strides中间两个为2 表示x,y方向都间隔1个取样

    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, img_pixel, img_pixel, channels])  # 彩色图像3个频道，如果是灰度图就是1

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])  # 图像60*60*32
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)  # 图像 30*30*32

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])  # 图像 30*30*64
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)  # 图像 15*15*64

        # Fully connected layer
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])  # [None,15*15*64]
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 3 input, 32 outputs 彩色图像3个输入(3个频道)，灰度图像1个输入
        'wc1': tf.Variable(tf.random_normal([5, 5, channels, 32])),  # 5X5的卷积模板
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([(1 + img_pixel // 4) * (1 + img_pixel // 4) * 64, 1024])),
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
    dBuf = np.ones((step, step), np.uint8) * 255  # 作为输出掩膜的像素

    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")

    srcDS = gdal.Open(dir_name, gdal.GA_ReadOnly)  # 只读方式打开原始影像
    geoTrans = srcDS.GetGeoTransform()  # 获取地理参考6参数
    srcPro = srcDS.GetProjection()  # 获取坐标引用
    srcXSize = srcDS.RasterXSize  # 宽度
    srcYSize = srcDS.RasterYSize  # 高度
    nbands = srcDS.RasterCount  # 波段数

    # raster_fn = path.join(dir_name, 'test_mask.tiff')  # 存放掩膜影像
    raster_fn = dir_name[0:-4] + "_mask_step" + str(step).zfill(2) + "_ds" + str(dBuf_size).zfill(2) + ".tiff"

    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, srcXSize, srcYSize, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geoTrans)  # 设置掩膜的地理参考
    target_ds.SetProjection(srcPro)  # 设置掩膜坐标引用
    band = target_ds.GetRasterBand(1)  # 获取第一个波段(影像只有一个波段)
    band.SetNoDataValue(0)  # 将这个波段的值全设置为0

    saver = tf.train.Saver()  # 默认是保存所有变量
    config = tf.ConfigProto(device_count={"CPU": 4},  # limit to num_cpu_core CPU usage
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=4,
                            log_device_placement=False)
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        flagY = True
        for i in range(0, srcYSize, step):
            if not flagY:
                break
            if i + isize > srcYSize - 1:
                i = srcYSize - 1 - isize
                flagY = False

            flagX = True
            for j in range(0, srcXSize, step):
                if not flagX:
                    break
                if j + isize > srcXSize - 1:
                    j = srcXSize - 1 - isize
                    flagX = False

                # multi band to array
                for band in range(nbands):
                    band += 1
                    # print "[ GETTING BAND ]: ", band
                    srcband = srcDS.GetRasterBand(band)  # 获取该波段
                    if srcband is None:
                        continue

                    # Read raster as arrays 类似RasterIO（C++）
                    dataraster = srcband.ReadAsArray(j, i, isize, isize, isize, isize)

                    if band == 1:
                        data = dataraster.reshape((isize, isize, 1))
                    else:
                        # 将每个波段的数组很并到一个3维数组中
                        data = np.append(data, dataraster.reshape((isize, isize, 1)), axis=2)

                data = data * (1. / 255) - 0.5
                data = data.reshape([-1, img_size])
                pred1 = sess.run(pred, feed_dict={x: data, keep_prob: 1.})

                # if sess.run(tf.argmax(pred1, 1)):
                if np.argmax(pred1, 1):
                    target_ds.GetRasterBand(1).WriteArray(dBuf, j, i)
        target_ds.FlushCache()  # 将数据写入文件
        endtime = datetime.datetime.now()
        print("endtime: ", endtime)
        print("time used in seconds: ", (endtime - starttime).seconds)


if __name__ == '__main__':
    main(sys.argv)
