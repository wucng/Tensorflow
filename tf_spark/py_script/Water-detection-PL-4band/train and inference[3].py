# -*- coding: UTF-8 -*-

'''
train=1 训练
train=-1 推理
'''

from __future__ import print_function
import numpy as np
import pickle
import os
from os import path
import zlib
import gzip
# import gdal
import sys
# import cv2
import datetime
import tensorflow as tf
# from numba.decorators import jit
# import multiprocessing
# from PIL import Image
from osgeo import gdal, ogr
from osgeo.gdalconst import *
# import glob


start_time = datetime.datetime.now()
print("startTime: ", start_time)

def create_pickle_train(image_path, mask_path, img_pixel=9, channels=4):
    m = 0

    image_data = Multiband2Array(image_path)
    print("data_matrix_max= ", image_data.max())
    print("data_matrix_min= ", image_data.min())
    # mask_data = cv2.split(cv2.imread(mask_path))[0] / 255
    mask_data=Multiband2Array(mask_path)/255

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
            train_label = mask_data[i+img_pixel//2,j+img_pixel//2]
            data2 = np.append(data1, train_label)[np.newaxis, :]  # 数据+标签

            data_list.append(data2)

            m += 1

            if m % 10000 == 0:
                print(datetime.datetime.now(), "compressed {number} images".format(number=m))
                data_matrix = np.array(data_list, dtype=int)

                data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * channels + 1)))
                return data_matrix

    print(len(data_list))
    print(m)

    data_matrix = np.array(data_list, dtype=int)

    data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * channels+1)))

    """
    with gzip.open('D:/train_data_64.pkl', 'ab') as writer:  # 以压缩包方式创建文件，进一步压缩文件
        pickle.dump(data_matrix, writer)  # 数据存储成pickle文件
    """
    return data_matrix # shape [none,9*9*4+1]

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

def next_batch(data, batch_size, flag, img_pixel=9, channels=4):
    global start_index  # 必须定义成全局变量
    global second_index  # 必须定义成全局变量

    if 1==flag:
        start_index = 0
    # start_index = 0
    second_index = start_index + batch_size

    if second_index > len(data):
        second_index = len(data)

    data1 = data[start_index:second_index]
    # print('start_index', start_index, 'second_index', second_index)

    start_index = second_index
    if start_index >= len(data):
        start_index = 0

    # 将每次得到batch_size个数据按行打乱
    index = [i for i in range(len(data1))]  # len(data1)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    data1 = data1[index]

    # 提取出数据和标签
    img = data1[:, 0:img_pixel * img_pixel * channels]

    # img = img * (1. / img.max) - 0.5
    img = img * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
    img = img.astype(np.float32)  # 类型转换

    label = data1[:, -1]
    label = label.astype(int)  # 类型转换

    return img, label

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    # 从标量类标签转换为一个one-hot向量
    num_labels = labels_dense.shape[0]        #label的行数
    index_offset = np.arange(num_labels) * num_classes
    # print index_offset
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

tarin=-1 # 1 tarin ;-1 inference
# ---------------------------------------------#
if tarin==1:
    image_path = r"E:\项目\AI-Waterdetection-PL-4band\train_data\train_img_01.tif"
    mask_path = r"E:\项目\AI-Waterdetection-PL-4band\train_data\train_img_mask_01.tif"
    data=create_pickle_train(image_path, mask_path, img_pixel=9, channels=4)

    img_nums = data.shape[0]
    print(img_nums)
isize = 9
img_channel = 4
img_pixel = isize
model_save_path = r"./model_walter/"

# Parameters
training_epochs = 2
batch_size = 100

display_step = 1
channels = img_channel

# Network Parameters
img_size = isize * isize * channels
label_cols = 2
dropout = 0.75


# -------------------------------------------------------------#
if tarin==1:
    # ---------设置动态学习效率
    # Constants describing the training process.
    # MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = batch_size  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.95  # Learning rate decay factor.
    INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.
    global_step = tf.Variable(0)

    # global_step = training_epochs * (img_nums // batch_size)  # Integer Variable counting the number of training steps     # //是整数除法
    # Variables that affect learning rate.
    num_batches_per_epoch = int(img_nums / batch_size)
    # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    decay_steps = int(num_batches_per_epoch * 10)
    # decay_steps = int(2)
    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                               global_step,
                                               decay_steps,
                                               LEARNING_RATE_DECAY_FACTOR,
                                               staircase=True)
if tarin==-1:
    global_step = tf.Variable(0)
    learning_rate=1e-5

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


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, img_pixel, img_pixel, channels])  # 9x9x4

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])  # 图像9x9x64
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)  # 图像 5x5x64

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])  # 图像 5x5x128
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)  # 图像 3x3x128

    # Convolution Layer
    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])  # 图像 3x3x256
    # Max Pooling (down-sampling)
    # conv3 = maxpool2d(conv3, k=2)  # 图像 2x2x256

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])  # [None,2*2*256]-->[None,1024]
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
    'wc1': tf.Variable(tf.random_normal([5, 5, channels, 64])),  # 5X5的卷积模板   #tf.random_normal是从正态分布中输出随机数
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    # 5x5 conv, 128 inputs, 128 outputs
    # 'wc3': tf.Variable(tf.random_normal([5, 5, 128, 256])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([(1+img_pixel // 4) * (1+img_pixel // 4) * 128, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, label_cols]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    # 'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([label_cols]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的op
init = tf.global_variables_initializer()


if __name__ == '__main__':

    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)  # 创建输出文件目录
    model_fn = path.join(model_save_path, 'save_net.ckpt')


    saver = tf.train.Saver()  # 默认是保存所有变量

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

        sess.run(init)

        # 验证之前是否已经保存了检查点文件
        ckpt = tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("提取现有模型训练")

        if tarin==1: # train
            total_batch = int(img_nums / batch_size)
            learning_rate_me = 0
            lrc = 0.0

            for epoch in range(training_epochs):
                np.random.shuffle(data)

                avg_cost = 0.
                flag = 1

                for i in range(total_batch):
                    img, label = next_batch(data, batch_size, flag, img_pixel=isize, channels=img_channel)
                    flag = 0
                    batch_xs = img.reshape([-1, img_size])

                    batch_ys = dense_to_one_hot(label[:, np.newaxis], label_cols)  # 生成多列标签   问题6，生成多列标签是干什么呢？   by xjxf
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c, p,gl_step = sess.run([optimizer, cost, pred,global_step], feed_dict={x: batch_xs,
                                                                  y: batch_ys, keep_prob: dropout})

                    if i%10==0:print('global_step',gl_step,'cost',c)
                    # Compute average loss
                    avg_cost += c / total_batch
                    learning_rate_me=sess.run(learning_rate)
                    if learning_rate_me!=lrc:
                        lrc=learning_rate_me
                        print("learning_rate:",str(learning_rate_me))

                # Display logs per epoch step
                if (epoch + 1) % display_step == 0:
                    print("time: ", datetime.datetime.now())
                    print("Epoch:", '%04d' % (epoch + 1),
                          "cost:", "{:.9f}".format(avg_cost),
                          'accuracy:', sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0}))
                    save_path = saver.save(sess, model_fn)  # model_fn 路径中不能出现中文否则会报错
                    print('Saver path:', save_path)
                    print("模型保存完毕！")
            print("Optimization Finished!")

            end_time = datetime.datetime.now()
            print("end_time: ", end_time)
            print("time used: ", end_time-start_time)

        if tarin==-1: # inference

            # dir_name_1 = './test/'
            # img_paths = glob.glob(dir_name_1 + '*.tif')  # 得到所有.tif文件路径

            img_paths = []
            # [img_paths.append('gtdata:////users/xiaoshaolin/tensorflow/hubei/' + i.replace('\n', ''))
            #  for i in open('./tile.log')]  # 从hdfs上读取文件tile.log
            [img_paths.append('HDFS://dm01-08-01.tjidc.dcos.com:8020/user/root/hubei_title/' + i.replace('\n', ''))
             for i in open('./tile.log')]  # 从hdfs上读取文件tile.log


            # isize = img_pixel  # 10 x 10的样本
            m = 1
            isizes = m * isize
            img_size = img_pixel * img_pixel * channels

            # 为了支持中文路径，请添加下面这句代码
            gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")

            gdal.AllRegister()  # 注册驱动
            ogr.RegisterAll()

            for dir_name in img_paths[:10]:

                srcDS = gdal.Open(dir_name, GA_ReadOnly)  # 只读方式打开原始影像

                geoTrans = srcDS.GetGeoTransform()  # 获取地理参考6参数
                srcPro = srcDS.GetProjection()  # 获取坐标引用
                srcXSize = srcDS.RasterXSize  # 宽度
                srcYSize = srcDS.RasterYSize  # 高度
                nbands = srcDS.RasterCount  # 波段数

                # raster_fn = path.join(dir_name, 'test_mask.tiff')  # 存放掩膜影像
                # raster_fn = dir_name + "_mask.tif"
                # raster_fn ="test_mask_1.tiff"
                raster_fn = './image_walter/'
                if not os.path.exists(raster_fn): os.makedirs(raster_fn)
                raster_fn += dir_name.split('/')[-1].split('.')[0] + "_mask.tif"

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
                # ckpt = tf.train.get_checkpoint_state(logdir)
                # if ckpt and ckpt.model_checkpoint_path:
                #     saver.restore(sess, ckpt.model_checkpoint_path)

                # 一次性读取图像所有数据
                for band in range(nbands):
                    # band=int(band)
                    band += 1
                    # print "[ GETTING BAND ]: ", band
                    srcband = srcDS.GetRasterBand(band)  # 获取该波段
                    if srcband is None:
                        continue

                    # Read raster as arrays 类似RasterIO（C++）
                    dataraster = srcband.ReadAsArray(0, 0, srcXSize, srcYSize, srcXSize,
                                                     srcYSize)  # .astype(np.uint8)  # 像素值转到[0~255]

                    if band == 1:
                        data = dataraster.reshape((srcYSize, srcXSize, 1))
                    else:
                        # 将每个波段的数组很并到一个3s维数组中
                        data = np.append(data, dataraster.reshape((srcYSize, srcXSize, 1)), axis=2)
                    del dataraster, srcband
                srcDS = None
                # data_1 = data * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
                data_1 = data


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

                        # max_ = np.reshape(np.max(data_2_1, 1), [np.shape(data_2_1)[0], 1])
                        # min_ = np.reshape(np.min(data_2_1, 1), [np.shape(data_2_1)[0], 1])
                        # data_2_1 = (data_2_1 - min_) / (max_ - min_)

                        data_2_1 = data_2_1 * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5

                        # """
                        # 10x10像素预测

                        for row, pred1 in enumerate(sess.run(pred, feed_dict={x: data_2_1, keep_prob: 1.})):
                            if np.argmax(pred1):
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

                compute(0, srcYSize)
                """
                try:
                    jit(compute(0, srcYSize), target='gpu')
                except:
                    jit(compute(0, srcYSize))
                """
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
