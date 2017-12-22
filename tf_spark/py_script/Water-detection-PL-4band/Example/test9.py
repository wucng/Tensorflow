#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
from os import path
import tensorflow as tf
import numpy as np
import datetime
from osgeo import gdal
import cv2
import sys


starttime = datetime.datetime.now()
print(starttime)
# 训练集文件路径
step = 1
# dir_name = 'temp.tif'
step = 1
append_num=5
training_epochs = 50
batch_size = 920
isize = 9
channels = 4


def Multiband2Array(path):
    src_ds = gdal.Open(path)
    if src_ds is None:
        print('Unable to open %s' % path)
        sys.exit(1)

    src_ds_array = src_ds.ReadAsArray()
    # c1 = src_ds_array[0, :, :]
    # c2 = src_ds_array[1, :, :]
    # c3 = src_ds_array[2, :, :]
    # c4 = src_ds_array[3, :, :]
    c1 = src_ds_array[0, :, :]
    c1 = np.lib.pad(c1, append_num, 'symmetric')
    c2 = src_ds_array[1, :, :]
    c2 = np.lib.pad(c2, append_num, 'symmetric')
    c3 = src_ds_array[2, :, :]
    c3 = np.lib.pad(c3, append_num, 'symmetric')
    c4 = src_ds_array[3, :, :]
    c4 = np.lib.pad(c4, append_num, 'symmetric')

    data = cv2.merge([c1, c2, c3, c4])

    return data


def return_list(image_path, img_pixel=9, channels=4):
    m = 0
    image_data = Multiband2Array(image_path)
    x_size, y_size = image_data.shape[:2]
    data_list = []
    for i in range(append_num-isize//2, x_size - img_pixel//2-append_num):
        if i + img_pixel >= x_size+img_pixel//2-append_num:
            i = x_size - append_num -img_pixel//2 - 1
        for j in range(append_num-isize//2, y_size - img_pixel//2-append_num):
            if j + img_pixel >= y_size+img_pixel//2-append_num:
                j = y_size - append_num-img_pixel//2 - 1
            cropped_data = image_data[i:i + img_pixel, j:j + img_pixel]
            data1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))
            data_list.append(data1)
            m += 1
    # for i in range(0, x_size - img_pixel + 1):
    #     if i + img_pixel > x_size:
    #         i = x_size - img_pixel - 1
    #     for j in range(0, y_size - img_pixel + 1):
    #         if j + img_pixel > y_size:
    #             j = y_size - img_pixel - 1
    #         cropped_data = image_data[i:i + img_pixel, j:j + img_pixel]
    #         data1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))
    #         data_list.append(data1)
    #         m += 1
            # if m % 1000000 == 0:
            #     print(datetime.datetime.now(), "compressed {number} images".format(number=m))
    print(m)
    return i+1-(append_num-isize//2), j+1-(append_num-isize//2), data_list

# time格式化
print("startTime: ", starttime)

'''
# CNN 完整程序  测试模型
'''
training_epochs = 50
batch_size = 920
isize = 9
channels = 4
global_step = tf.Variable(0)

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
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)  # strides中间两个为1 表示x,y方向都不间隔取样
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')  # strides中间两个为2 表示x,y方向都间隔1个取样

img_pixel = isize


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
    # Reshape conv2 output to fit fully connected layer input
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
    'wc1': tf.Variable(tf.random_normal([5, 5, channels, 64])),  # 5X5的卷积模板
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([(1 + img_pixel // 4) * (1 + img_pixel // 4) * 128, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, label_cols]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([label_cols]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

dBuf = np.ones((step, step), np.uint8) * 255  # 作为输出掩膜的像素颗粒


def test(argv):
    # 参数设置  __start
    model_path = argv[1] + "/save_net.ckpt"  # 模型路径
    dir_name = argv[2]  # 影像路径
    step = int(argv[3])  # 识别步长，也直接影响填充块大小
    # 为了支持中文路径，请添加下面这句代码
    batch_size=500

    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")

    srcDS = gdal.Open(dir_name, gdal.GA_ReadOnly)  # 只读方式打开原始影像

    geoTrans = srcDS.GetGeoTransform()  # 获取地理参考6参数
    srcPro = srcDS.GetProjection()  # 获取坐标引用
    srcXSize = srcDS.RasterXSize  # 宽度
    srcYSize = srcDS.RasterYSize  # 高度
    nbands = srcDS.RasterCount  # 波段数
    print("nbands = :", nbands)

    h_size, w_size, date_list = return_list(dir_name, img_pixel=isize, channels=channels)
    print(h_size, w_size, len(date_list))

    shortname, extension = path.splitext(dir_name)
    raster_fn = shortname + "_mask_step" + str(step).zfill(2) + ".tiff"
    blur_fn = shortname + "_mask_step_" + str(step).zfill(2) + ".tiff"

    saver = tf.train.Saver()  # 默认是保存所有变量
    config = tf.ConfigProto(device_count={"CPU": 4},  # limit to num_cpu_core CPU usage
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=4,
                            log_device_placement=False)
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)

        result = []
        n = 0
        for i in range(0, len(date_list), batch_size):
            if i + batch_size > len(date_list):
                batch_size = len(date_list) - i
            cropped_data = np.array(date_list[i:i+batch_size]) / 255.0-0.5
            cropped_data = cropped_data.reshape([-1, img_size])
            pred1 = sess.run(pred, feed_dict={x: cropped_data, keep_prob: 1.})
            pred1 = np.argmax(pred1, 1)
            result.extend(pred1)
        result = np.array(result)
        img = np.reshape(result, [int(len(date_list)/w_size), w_size])
        img=img*255
        # img = np.lib.pad(img * 255, ((4, 4), (4, 4)), 'constant', constant_values=0)
        cv2.imwrite(raster_fn, img)

        # 添加坐标
        target_ds = gdal.Open(raster_fn, gdal.GA_Update)
        target_ds.SetGeoTransform(geoTrans)
        target_ds.SetProjection(srcPro)
        target_ds.FlushCache()

        # 滤波
        im = cv2.imread(raster_fn)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(im, 5)
        cv2.imwrite(blur_fn, blur)

        target_ds = gdal.Open(blur_fn, gdal.GA_Update)
        target_ds.SetGeoTransform(geoTrans)
        target_ds.SetProjection(srcPro)
        target_ds.FlushCache()

        endtime = datetime.datetime.now()
        print("endtime: ", endtime)
        print("time used in seconds: ", (endtime - starttime).seconds)


if __name__ == '__main__':
    test(sys.argv)