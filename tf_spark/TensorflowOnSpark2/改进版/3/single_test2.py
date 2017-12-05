# -*- coding:utf-8 -*-

"""
2x2 单机测试模型精度（有全连接层），对应mnist_dist.py
IMAGE_PIXELS=1 对应1x1
IMAGE_PIXELS=2 对应2x2

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gzip
import pickle
import os
import datetime

IMAGE_PIXELS=2
# IMAGE_PIXELS=10 # 图像大小 mnist 28x28x1  (后续参考自己图像大小进行修改)
channels=3
num_class=2
global dropout
dropout = 0.7
global learning_rate
learning_rate=1e-3
init_lr=1e-2
batch_size=200
train=1  # 0 test 1 train -1 inference
logdir='./model_arable/'
acc1=0.8 # 阈值
if train==0 or train==1:img_path='./11_0.pkl'
flag=True

if train==-1:
    dir_name = "./11.tif"
    isize = IMAGE_PIXELS  # 10 x 10的样本
    m = 1
    isizes = m * isize
    img_size=IMAGE_PIXELS*IMAGE_PIXELS*channels
# --------------------------------------


def read_and_decode_1(filename):
    with gzip.open(filename, 'rb') as pkl_file:  # 打开文件
        data = pickle.load(pkl_file)  # 加载数据

    return data
# --------------------------------------


def dense_to_one_hot2(labels_dense,num_classes):
    labels_dense=np.array(labels_dense,dtype=np.uint8)
    num_labels = labels_dense.shape[0] # 标签个数
    labels_one_hot=np.zeros((num_labels,num_classes),np.uint8)
    for i,itenm in enumerate(labels_dense):
        labels_one_hot[i,itenm]=1
    return labels_one_hot

start_index=0
def next_batch(data,batch_size,img_pixel=60,channels=4):
    global start_index  # 必须定义成全局变量
    global second_index  # 必须定义成全局变量

    second_index=start_index+batch_size
    if second_index>len(data):
        second_index=len(data)
    data1=data[start_index:second_index]
    # lab=labels[start_index:second_index]
    start_index=second_index
    if start_index>=len(data):
        start_index = 0

    # 将每次得到batch_size个数据按行打乱
    index = [i for i in range(len(data1))]  # len(data1)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    data1 = data1[index]

    # 提起出数据和标签
    img = data1[:, 0:img_pixel * img_pixel * channels]

    # Z-score标准化方法
    # mean = np.reshape(np.average(img, 1), [np.shape(img)[0], 1])
    # std = np.reshape(np.std(img, 1), [np.shape(img)[0], 1])
    #
    # img = (img - mean) / std
    # min-max标准化（Min-Max Normalization
    max_=np.reshape(np.max(img,1),[np.shape(img)[0], 1])
    min_ = np.reshape(np.min(img, 1), [np.shape(img)[0], 1])

    img=(img-min_)/(max_-min_)
    # img = img * (1. / np.max(img)) - 0.5
    # img = img * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
    # img=img.astype(float) # 类型转换

    label = data1[:, -1]
    label = label.astype(int)  # 类型转换

    return img,label
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


if IMAGE_PIXELS==2: #2x2像素
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 3 input, 32 outputs 彩色图像3个输入(3个频道)，灰度图像1个输入
        'wc1': tf.get_variable('wc1',[3,3,channels,64],dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),  # 5X5的卷积模板

        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.get_variable('wc2',[3,3,32,64],dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),

        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([(IMAGE_PIXELS // 2) * (IMAGE_PIXELS // 2) * 64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, num_class]))
    }

    biases = {
        'bc1': tf.get_variable('bc1',[64],dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
        'bc2': tf.get_variable('bc2',[64],dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([num_class]))
    }

    # Placeholders or QueueRunner/Readers for input data
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS * channels], name="x")  # mnist 28*28*1
    y_ = tf.placeholder(tf.float32, [None, num_class], name="y_")
    keep=tf.placeholder(tf.float32)

    x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, channels])  # mnist 数据 28x28x1 (灰度图 波段为1)
    # x_img=tf.image.resize_image_with_crop_or_pad(x_img,IMAGE_PIXELS,IMAGE_PIXELS) # shape [N,10,10,3]
    # tf.summary.image("x_img", x_img)

    # 改成卷积模型
    conv1 = conv2d(x_img, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2) # shape [N,1,1,32]
    fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1=tf.nn.relu(fc1)
    fc1=tf.nn.dropout(fc1,keep)
    y=tf.add(tf.matmul(fc1, weights['out']), biases['out'])

if IMAGE_PIXELS==1: # 1x1 像素
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 3 input, 32 outputs 彩色图像3个输入(3个频道)，灰度图像1个输入
        'wc1': tf.get_variable('wc1', [3, 3, channels, 32], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),  # 5X5的卷积模板

        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.get_variable('wc2', [3, 3, 32, 64], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),

        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([(1+IMAGE_PIXELS // 2) * (1+IMAGE_PIXELS // 2) * 32, num_class])),
        'wd2': tf.Variable(tf.random_normal([IMAGE_PIXELS * IMAGE_PIXELS * channels, num_class])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, num_class]))
    }

    biases = {
        'bc1': tf.get_variable('bc1', [32], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
        'bc2': tf.get_variable('bc2', [64], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
        'bd1': tf.Variable(tf.random_normal([num_class])),
        'bd2': tf.Variable(tf.random_normal([num_class])),
        'out': tf.Variable(tf.random_normal([num_class]))
    }

    # Placeholders or QueueRunner/Readers for input data
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS * channels], name="x")  # mnist 28*28*1
    y_ = tf.placeholder(tf.float32, [None, num_class], name="y_")
    keep = tf.placeholder(tf.float32)


    x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, channels])  # mnist 数据 28x28x1 (灰度图 波段为1)
    # x_img=tf.image.resize_image_with_crop_or_pad(x_img,IMAGE_PIXELS,IMAGE_PIXELS) # shape [N,10,10,3]
    # tf.summary.image("x_img", x_img)
    # """
    y = tf.add(tf.matmul(x, weights['wd2']), biases['bd2'])
    y=tf.sigmoid(y)

    """
    # 改成卷积模型
    conv1 = conv2d(x_img, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)  # shape [N,1,1,32]
    # conv1 = tf.nn.dropout(conv1, keep)
    fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
    y = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    """

"""
conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
conv2 = maxpool2d(conv2, k=2)

fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
fc1 = tf.nn.relu(fc1)
# if args.mode == "train" or args.mode == "retrain":
fc1 = tf.nn.dropout(fc1, keep)
y = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
"""
global_step = tf.Variable(0, name="global_step", trainable=False)

if train==0 or train==1:
    data=read_and_decode_1(img_path)
    np.random.shuffle(data) # 随机打乱
# #--------------学习速率的设置（学习速率呈指数下降）--------------------- #将 global_step/decay_steps 强制转换为整数
# learning_rate = tf.train.exponential_decay(init_lr,global_step, decay_steps=100, #decay_steps=len(data)//batch_size,
#                                            decay_rate=0.98,staircase=True)


# loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# tf.summary.scalar("loss", loss)
# tf.train.GradientDescentOptimizer()
# train_op = tf.train.AdagradOptimizer(learning_rate).minimize(
#     loss, global_step=global_step)
train_op = tf.train.AdagradOptimizer(learning_rate).minimize(
    loss, global_step=global_step)


# Test trained model
label = tf.argmax(y_, 1, name="label")
prediction = tf.argmax(y, 1, name="prediction")
correct_prediction = tf.equal(prediction, label)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
# tf.summary.scalar("acc", accuracy)

saver = tf.train.Saver()

# summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()


if __name__=="__main__":
    with tf.Session() as sess:
        if train==1:
            # 验证之前是否已经保存了检查点文件
            ckpt = tf.train.get_checkpoint_state(logdir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            # if os.path.exists('model_aralbe/checkpoint'):
            #     saver.restore(sess, 'model_aralbe/model.ckpt-9999')
            else:
                sess.run(init_op)
            n=0
            for step in range(10000):
                if not step%500:np.random.shuffle(data) # 随机打乱
                batch_xs, batch_ys = next_batch(data, batch_size, img_pixel=IMAGE_PIXELS, channels=3)
                batch_ys = dense_to_one_hot2(batch_ys, 2)
                feed = {x: batch_xs, y_: batch_ys,keep:dropout}
                sess.run(train_op,feed_dict=feed)
                acc = sess.run(accuracy, {x: batch_xs, y_: batch_ys, keep: 1.})
                if acc > acc1:
                    if flag and acc>0.9:
                        os.popen('rm -rf ' + logdir + '*')  # 删除路径下的所有文件、文件夹
                        flag=False
                    # acc1=acc # 训练达到一定程度加上
                    saver.save(sess, logdir+'model.ckpt', global_step=step)
                    n=0
                else:
                    n+=1
                    if n>10:
                        ckpt1 = tf.train.get_checkpoint_state(logdir)
                        if ckpt1 and ckpt1.model_checkpoint_path:
                            saver.restore(sess, ckpt1.model_checkpoint_path)
                        if learning_rate > 1e-7:
                            learning_rate = learning_rate * .8
                            # learning_rate = learning_rate * .8**(step/10)
                        else:
                            learning_rate = 1e-3
                        if dropout>0.2:
                            dropout=dropout*.85
                        else:
                            dropout = .7

                if step%100==0:
                    [loss1]=sess.run([loss],feed)
                    print("step",step,"loss",loss1,"acc",acc)

        elif train==0:
            ckpt = tf.train.get_checkpoint_state(logdir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            batch_xs, batch_ys = next_batch(data,50000,img_pixel=IMAGE_PIXELS,channels=3)
            batch_ys=dense_to_one_hot2(batch_ys,2)
            feed = {x: batch_xs, y_: batch_ys,keep:1.}
            acc = sess.run(accuracy, feed_dict=feed)
            print("acc: {0}".format(acc))

        else: # train==-1
            from numba.decorators import jit
            # import multiprocessing
            # from PIL import Image
            from osgeo import gdal, ogr
            from osgeo.gdalconst import *


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
            raster_fn = dir_name + "_mask.tif"
            # raster_fn ="test_mask_1.tiff"

            target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, srcXSize, srcYSize, 1, gdal.GDT_Byte)
            target_ds.SetGeoTransform(geoTrans)  # 设置掩膜的地理参考
            target_ds.SetProjection(srcPro)  # 设置掩膜坐标引用
            # band = target_ds.GetRasterBand(1)  # 获取第一个波段(影像只有一个波段)
            # band.SetNoDataValue(0)  # 将这个波段的值全设置为0
            target_ds.GetRasterBand(1).SetNoDataValue(0)

            del geoTrans
            del srcPro
            del dir_name, raster_fn

            dBuf_1 = np.zeros([srcYSize, srcXSize], np.uint8)  # 整个掩膜的缓存

            # saver = tf.train.Saver()  # 默认是保存所有变量
            # cdef int i,j,band,xx,yy

            start = datetime.datetime.now()
            print('session start:', datetime.datetime.now())
            ckpt = tf.train.get_checkpoint_state(logdir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # 一次性读取图像所有数据
            for band in range(nbands):
                # band=int(band)
                band += 1
                # print "[ GETTING BAND ]: ", band
                srcband = srcDS.GetRasterBand(band)  # 获取该波段
                if srcband is None:
                    continue

                # Read raster as arrays 类似RasterIO（C++）
                dataraster = srcband.ReadAsArray(0, 0, srcXSize, srcYSize, srcXSize, srcYSize)  #.astype(np.uint8)  # 像素值转到[0~255]

                if band == 1:
                    data = dataraster.reshape((srcYSize, srcXSize, 1))
                else:
                    # 将每个波段的数组很并到一个3s维数组中
                    data = np.append(data, dataraster.reshape((srcYSize, srcXSize, 1)), axis=2)
                del dataraster, srcband
            srcDS = None
            # data_1 = data * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
            data_1=data
            def compute(YSize1, YSize2):  # 每次计算YSize1~YSize2 行，实现伪并行计算
                flagY = True

                for i in range(YSize1, YSize2, isizes):
                    # print("正在计算：%.2f%%" % (i/srcYSize * 100), end="\r")
                    if not flagY: break
                    if i + isizes > YSize2 - 1:
                        i = YSize2 - 1 - isizes
                        flagY = False

                    # print('loop start:', datetime.datetime.now())

                    flagX = True
                    for j in range(0, srcXSize, isizes):
                        if not flagX: break

                        if j + isizes > srcXSize - 1:
                            j = srcXSize - 1 - isizes
                            flagX = False

                        data2 = data_1[i:i + isize, j:j + isize].reshape([-1, img_size])

                        if j == 0:
                            data_2_1 = data2
                        else:
                            data_2_1 = np.vstack((data_2_1, data2))  # 取出一个isize行所有的image,再进行预测（而不是一张张预测）

                    # 标准化
                    # Z-score标准化方法
                    # mean = np.reshape(np.average(data_2_1, 1), [np.shape(data_2_1)[0], 1])
                    # std = np.reshape(np.std(data_2_1, 1), [np.shape(data_2_1)[0], 1])
                    #
                    # data_2_1 = (data_2_1 - mean) / std

                    max_ = np.reshape(np.max(data_2_1, 1), [np.shape(data_2_1)[0], 1])
                    min_ = np.reshape(np.min(data_2_1, 1), [np.shape(data_2_1)[0], 1])

                    data_2_1 = (data_2_1 - min_) / (max_ - min_)

                    # """
                    # 10x10像素预测
                    for row,pred in enumerate(sess.run(prediction,feed_dict={x: data_2_1, keep: 1.})):
                        if pred:
                            # print('pred3', pred3)
                            jj = row * isize
                            if jj + isize >= srcXSize - 1:
                                jj = srcXSize - 1 - isize
                            dBuf_1[i:i + isize, jj:jj + isize] = np.ones([isize, isize], np.uint8) * 255
                    # """

                    '''
                    # 逐像素 预测
                    for row, pred in enumerate(sess.run(prediction, feed_dict={x: data_2_2, keep: 1.})):
                        if pred:
                            jj = row * isize
                            if jj + isize >= srcXSize - 1:
                                jj = srcXSize - 1 - isize
                            # dBuf_1[i:i + isize, jj:jj + isize] = np.ones([isize, isize], np.uint8) * 255
                            data3=np.reshape(data_2_1[row],[isize,isize,channels])
                            list_d=[]
                            for iii in range(isize):
                                for jjj in range(isize):
                                    data4=tf.image.resize_image_with_crop_or_pad(np.reshape(data3[iii,jjj,:],[1,1,channels]),isize,isize) # 10x10x3
                                    data4=tf.reshape(data4,[-1,img_size])
                                    list_d.append(data4)
                            list_d=tf.reshape(list_d,[-1,img_size])
                            data5_1=sess.run(list_d)
                            # 标准化
                            max_ = np.reshape(np.max(data5_1, 1), [np.shape(data5_1)[0], 1])
                            min_ = np.reshape(np.min(data5_1, 1), [np.shape(data5_1)[0], 1])
                            # if max_.any()!=min_.any():
                            data5_1 = (data5_1 - min_) / (max_ - min_+1.)
                            pred2=sess.run(prediction, feed_dict={x: data5_1, keep: 1.})
                            dBuf_1[i:i + isize, jj:jj + isize] = np.reshape(pred2,[isize,isize])*255
                    '''

            # 使用多线程,numba 加速

            # jit(compute(0, srcYSize))
            try:
                jit(compute(0,srcYSize),target='gpu')
            except:
                jit(compute(0, srcYSize))


            print(datetime.datetime.now() - start)
            print('writer data:', datetime.datetime.now())
            target_ds.GetRasterBand(1).WriteArray(dBuf_1, 0, 0)
            target_ds.FlushCache()  # 将数据写入文件
            target_ds = None
            print("计算完成：100.00%", end="\r")
            # endtime = datetime.datetime.now()
            print('end time:', datetime.datetime.now())
            # print((endtime - starttime).seconds)
            sess.close()
            exit()

