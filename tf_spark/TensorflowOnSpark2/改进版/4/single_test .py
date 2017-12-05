# -*- coding:utf-8 -*-

"""
10x10 标签对应10x10 单机测试模型精度（舍弃全连接层），对应mnist_dist.py
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
from tensorflow.contrib.layers.python.layers import batch_norm

IMAGE_PIXELS=16 # 图像大小 mnist 28x28x1  (后续参考自己图像大小进行修改)
channels=3
num_class=IMAGE_PIXELS*IMAGE_PIXELS
dropout = 0.7
learning_rate=1e-3
INITIAL_LEARNING_RATE=1e-2
init_lr=1e-2
batch_size=128
train=-1 # 0 test 1 train -1 inference
logdir='model_arable/'
acc1=0.8 # 阈值
if train==0 or train==1:img_path='11.pkl'
flag=True

if train==-1:
    dir_name = "11.tif"
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

def batch_norm_layer(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: batch_norm(inputT, is_training=True,
                                      center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                   lambda: batch_norm(inputT, is_training=False,
                                      center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                                      scope=scope))  # , reuse = True))


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
    #mean = np.reshape(np.average(img, 1), [np.shape(img)[0], 1])
    #std = np.reshape(np.std(img, 1), [np.shape(img)[0], 1])

    #img = (img - mean) / std
    # min-max标准化（Min-Max Normalization
    max_=np.reshape(np.max(img,1),[np.shape(img)[0], 1])
    min_ = np.reshape(np.min(img, 1), [np.shape(img)[0], 1])

    img=(img-min_)/(max_-min_+0.0001)
    # img = img * (1. / np.max(img)) - 0.5
    # img = img * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
    # img=img.astype(float) # 类型转换

    label = data1[:, img_pixel * img_pixel * channels:]
    label=np.reshape(label,[-1,img_pixel,img_pixel])
    label = label.astype(int)  # 类型转换

    return img,label
# Create some wrappers for simplicity
def conv2d(x, W, b,is_training=True,strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)  # strides中间两个为1 表示x,y方向都不间隔取样
    x=batch_norm_layer(x,is_training)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')  # strides中间两个为2 表示x,y方向都间隔1个取样

def maxpool2d2(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')  # strides中间两个为2 表示x,y方向都间隔1个取样

# Store layers weight & bias
weights = {
    'wc1': tf.get_variable('wc1',[3,4,channels,64],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss)*0.001,
    # 'wc2': tf.get_variable('wc2',[3,3,64,64],dtype=tf.float32,
    #                        initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss)*0.001,

    'wc3': tf.get_variable('wc3', [3, 3, 64, 128], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
    # 'wc4': tf.get_variable('wc4', [3, 3, 128, 128], dtype=tf.float32,
    #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,

    'wc5': tf.get_variable('wc5', [3, 3, 128, 256], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
    'wc6': tf.get_variable('wc6', [3, 3, 256, 256], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
    'wc7': tf.get_variable('wc7', [3, 3, 256, 256], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
    # 'wc8': tf.get_variable('wc8', [3, 3, 256, 256], dtype=tf.float32,
    #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,

    # 'wc9': tf.get_variable('wc9', [3, 3, 256, 256], dtype=tf.float32,
    #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
    # 'wc10': tf.get_variable('wc10', [3, 3, 256, 256], dtype=tf.float32,
    #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
    # 'wc11': tf.get_variable('wc11', [3, 3, 256, 256], dtype=tf.float32,
    #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
    # 'wc12': tf.get_variable('wc12', [3, 3, 256, 256], dtype=tf.float32,
    #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
}

biases = {
    'bc1': tf.get_variable('bc1',[64],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
    'bc2': tf.get_variable('bc2',[64],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),

    'bc3': tf.get_variable('bc3', [128], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
    'bc4': tf.get_variable('bc4',[128],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),

    'bc5': tf.get_variable('bc5',[256],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
    'bc6': tf.get_variable('bc6',[256],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
    'bc7': tf.get_variable('bc7', [256], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
    'bc8': tf.get_variable('bc8',[256],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),

    # 'bc9': tf.get_variable('bc9',[256],dtype=tf.float32,
    #                        initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
    # 'bc10': tf.get_variable('bc10',[256],dtype=tf.float32,
    #                        initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
    # 'bc11': tf.get_variable('bc11', [256], dtype=tf.float32,
    #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
    # 'bc12': tf.get_variable('bc12',[256],dtype=tf.float32,
    #                        initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
}

# Placeholders or QueueRunner/Readers for input data
x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS * channels], name="x")  # mnist 28*28*1
y_ = tf.placeholder(tf.float32, [None, IMAGE_PIXELS,IMAGE_PIXELS], name="y_")
keep=tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool, name='MODE')

x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, channels])  # mnist 数据 28x28x1 (灰度图 波段为1)
# tf.summary.image("x_img", x_img)

# 改成卷积模型
conv1 = conv2d(x_img, weights['wc1'], biases['bc1'],is_training)
# conv1=conv2d(conv1,weights['wc2'],biases['bc2'],is_training)
conv1=tf.nn.lrn(conv1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
conv1 = maxpool2d(conv1, k=2)
conv1 = tf.nn.dropout(conv1, keep)

conv2 = conv2d(conv1, weights['wc3'], biases['bc3'],is_training)
# conv2 = conv2d(conv2, weights['wc4'], biases['bc4'],is_training)
conv2=tf.nn.lrn(conv2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
conv2 = maxpool2d(conv2, k=2)
conv2 = tf.nn.dropout(conv2, keep)

conv3 = conv2d(conv2, weights['wc5'], biases['bc5'],is_training)
# conv3 = conv2d(conv3, weights['wc6'], biases['bc6'],is_training)
conv3=tf.nn.lrn(conv3, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
conv3 = maxpool2d(conv3, k=2)
conv3 = tf.nn.dropout(conv3, keep)

conv4 = conv2d(conv3, weights['wc7'], biases['bc7'],is_training)
# conv4 = conv2d(conv4, weights['wc8'], biases['bc8'],is_training)
conv4=tf.nn.lrn(conv4, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
conv4 = maxpool2d(conv4, k=2)

# conv4 = conv2d(conv4, weights['wc9'], biases['bc9'],is_training)
# conv4 = conv2d(conv4, weights['wc10'], biases['bc10'],is_training)
# conv4 = conv2d(conv4, weights['wc11'], biases['bc11'],is_training)
# conv4 = conv2d(conv4, weights['wc12'], biases['bc12'],is_training)
# conv4=tf.nn.lrn(conv4, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
# conv4 = maxpool2d2(conv4, k=2)
y = tf.reshape(conv4, [-1, num_class])

global_step = tf.Variable(0, name="global_step", trainable=False)

if train==0 or train==1:
    data=read_and_decode_1(img_path)
    np.random.shuffle(data) # 随机打乱
#--------------学习速率的设置（学习速率呈指数下降）---------------------
# #将 global_step/decay_steps 强制转换为整数
# learning_rate = tf.train.exponential_decay(init_lr,global_step, decay_steps=100, #decay_steps=len(data)//batch_size,
#                                            decay_rate=0.98,staircase=True)
# learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
      #                                            global_step,
      #                                            10000,
      #                                            0.96,
      #                                            staircase=False)
# learning_rate=tf.train.polynomial_decay(INITIAL_LEARNING_RATE,global_step,3000000,1e-5,0.8,False)

# loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

# loss=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y))

y=tf.reshape(y,[-1,IMAGE_PIXELS,IMAGE_PIXELS])

# loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-y_)))
loss=tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y))

# loss=tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(y,y_))))

# tf.summary.scalar("loss", loss)
# train_op = tf.train.AdagradOptimizer(learning_rate).minimize(
#     loss, global_step=global_step)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss, global_step=global_step)

# Test trained model
# label = tf.argmax(y_, 1, name="label")
# prediction = tf.argmax(y, 1, name="prediction")
# correct_prediction = tf.equal(prediction, label)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

# prediction=y
# tf.summary.scalar("acc", accuracy)
def compute_acc(xs,ys,IMAGE_PIXELS):
    global y
    y1 = sess.run(y,{x:xs,y_:ys,keep:1.,is_training:True})
    prediction = [1. if abs(x3 - 1) < abs(x3 - 0) else 0. for x1 in y1 for x2 in x1 for x3 in x2]
    prediction = np.reshape(prediction, [-1, IMAGE_PIXELS, IMAGE_PIXELS]).astype(np.uint8)
    # correct_prediction = np.equal(prediction, np.reshape(ys,[-1, IMAGE_PIXELS, IMAGE_PIXELS])).astype(tf.float32)
    # accuracy = np.mean(correct_prediction)
    accuracy=np.mean(np.equal(prediction, ys).astype(np.float32))
    # print(prediction[0,:,:])
    # print('-----------------')
    # print(ys[0,:,:])
    # exit()
    return accuracy

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
            for step in range(10):
                if not step%500:np.random.shuffle(data) # 随机打乱
                batch_xs, batch_ys = next_batch(data, batch_size, img_pixel=IMAGE_PIXELS, channels=channels)
                # batch_ys = dense_to_one_hot2(batch_ys, 2)
                feed = {x: batch_xs, y_: batch_ys,keep:dropout,is_training:True}
                sess.run(train_op,feed_dict=feed)

                # feed={x: batch_xs, y_: batch_ys, keep: 1., is_training: False}
                acc=compute_acc(batch_xs, batch_ys, IMAGE_PIXELS)

                # acc = sess.run(accuracy, {x: batch_xs, y_: batch_ys, keep: 1.,is_training:False})
                if acc > acc1:
                    if flag and acc>0.9:
                        os.popen('rm -rf ' + logdir + '*')  # 删除路径下的所有文件、文件夹
                        flag=False
                    # acc1=acc # 训练达到一定程度加上
                    saver.save(sess, logdir+'model.ckpt', global_step=step)
                    n=0
                else:
                    n += 1
                    if n > 100:
                        ckpt1 = tf.train.get_checkpoint_state(logdir)
                        if ckpt1 and ckpt1.model_checkpoint_path:
                            saver.restore(sess, ckpt1.model_checkpoint_path)
                        if learning_rate > 1e-7:
                            learning_rate = learning_rate * .8
                            # learning_rate = learning_rate * .8**(step/10)
                        else:
                            learning_rate = 1e-3
                        if dropout > 0.2:
                            dropout = dropout * .85
                        else:
                            dropout = .7

                if step%5==0:
                    [loss1]=sess.run([loss],feed)
                    print("step",step,"loss",loss1,"acc",acc)
                    # print("step", step, "loss", loss1)
                    saver.save(sess, logdir + 'model.ckpt', global_step=step)

        elif train==0:
            ckpt = tf.train.get_checkpoint_state(logdir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            batch_xs, batch_ys = next_batch(data,1000,img_pixel=IMAGE_PIXELS,channels=channels)
            # batch_ys=dense_to_one_hot2(batch_ys,2)
            # feed = {x: batch_xs, y_: batch_ys,keep:1.,is_training: False}
            # acc = sess.run(accuracy, feed_dict=feed)
            acc = compute_acc(batch_xs, batch_ys, IMAGE_PIXELS)
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
                    max_ = np.reshape(np.max(data_2_1, 1), [np.shape(data_2_1)[0], 1])
                    min_ = np.reshape(np.min(data_2_1, 1), [np.shape(data_2_1)[0], 1])

                    data_2_1 = (data_2_1 - min_) / (max_ - min_+0.0001)

                    # """
                    # 10x10像素预测
                    prediction=sess.run(y,feed_dict={x: data_2_1, keep: 1.,is_training:True})
                    prediction = [1. if abs(x3 - 1) < abs(x3 - 0) else 0. for x1 in prediction for x2 in x1 for x3 in x2]
                    prediction = np.reshape(prediction, [-1, IMAGE_PIXELS, IMAGE_PIXELS]).astype(np.uint8)

                    # print(prediction[0,:,:])
                    # exit()

                    for row,pred in enumerate(prediction):
                        # pred1=[1 if ix>=1./num_class else 0 for ix in pred]
                        jj = row * isize
                        if jj + isize >= srcXSize - 1:
                            jj = srcXSize - 1 - isize
                        dBuf_1[i:i + isize, jj:jj + isize] = np.reshape(pred,[isize,isize])*255

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
            exit()

