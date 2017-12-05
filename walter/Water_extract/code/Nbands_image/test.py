#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import os
from os import path
import tensorflow as tf
from PIL import Image
import numpy as np
import tool_set
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import datetime

starttime = datetime.datetime.now()
# 训练集文件路径
# dir_name='/home/ubuntu_wu/桌面/image_60/image3/image2/'
# dir_name='G:/datas/test4_datas/'
# dir_name='F:/PL/test_data/'
# dir_name='F:/PL/2/test_data/'
dir_name='F:/PL/20170321_022817_0e0d_visual/test_data/'
'''
# CNN 完整程序  测试模型
'''
# Parameters
# learning_rate = 10**(-5)
training_epochs = 500
# training_iters = 200000
batch_size = 128
display_step = 10
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

#初始化所有的op
# init = tf.global_variables_initializer()


if __name__ == '__main__':

    # 提起pickle数据 data包含 特征+标签
    data = tool_set.read_and_decode(dir_name + "test_data.pkl", img_pixel, channels)

    # # 打乱数据  这里不打乱 否则测试时 数据与图像名对应不起来
    # index = [i for i in range(len(data))]  # len(data)得到的行数
    # np.random.shuffle(index)  # 将索引打乱
    # data = data[index]

    # 提起出数据和标签
    img = data[:, 0:img_pixel * img_pixel * channels]

    # img = img * (1. / img.max) - 0.5
    img = img * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
    # img=img.astype(float) # 类型转换

    label = data[:, -1]
    label = label.astype(int)  # 类型转换


    # 打开记录文件名的pickle文件
    with open(dir_name+'images_name.pkl', 'rb') as pkl_file:
        img_names = pickle.load(pkl_file) # 加载图像名列表

    saver = tf.train.Saver()  # 默认是保存所有变量

    with tf.Session() as sess:
        # sess.run(init) # 提起模型数据不能在执行变量初始化

        # saver.restore(sess, "G:/datas2/model/save_net.ckpt")  # 提取模型数据

        # saver.restore(sess, "F:/PL/2/train_data/model/save_net.ckpt")  # 提取模型数据

        # saver.restore(sess, "F:/PL/model/save_net.ckpt")  # 提取模型数据
        saver.restore(sess, r"F:\PL\20170321_022817_0e0d_visual\train_data\model\save_net.ckpt")  # 提取模型数据



        X_test = img.reshape([-1, img_size])
        y_test = tool_set.dense_to_one_hot(label[:, np.newaxis], label_cols)

        print("Accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.}))

        # 预览预测结果
        pred1 = sess.run(pred, feed_dict={x: X_test[80:100], keep_prob: 1.})

        print(sess.run(tf.nn.softmax(pred1)))

        # tf.argmax(pred, 1)

        print(img_names[80:100])

    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
