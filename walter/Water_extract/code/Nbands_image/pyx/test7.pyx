#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
# import os
# import os
# from os import path
import tensorflow as tf
# from PIL import Image
import numpy as np

import pyximport
pyximport.install()
import tool_set
# import pandas as pd

# 训练集文件路径
# dir_name='/home/ubuntu_wu/桌面/image_60/image3/image2/'
# dir_name='F:/PL/train_data/'
# dir_name_1='F:/PL/2/train_data/'
dir_name='/home/wu/Water_extract/data/data/'
'''
# CNN 完整程序  训练模型
'''
# Parameters
# learning_rate = 10**(-5)
cdef int training_epochs = 500
# training_iters = 200000
cdef int batch_size = 128
cdef int display_step = 10
cdef int img_pixel=10
cdef int channels=4

# Network Parameters
cdef int img_size = img_pixel*img_pixel*channels # data input (img shape: 28*28*3)
cdef int label_cols = 2 # total classes (云、非云)使用标签[1,0,0] 3维
cdef float dropout = 0.75 # Dropout, probability to keep units
cdef int img_nums=15000


# ---------设置动态学习效率
# Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
cdef int NUM_EPOCHS_PER_DECAY = batch_size      # Epochs after which learning rate decays.
cdef float LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
cdef float INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

cdef int global_step=training_epochs*(img_nums//batch_size)   # Integer Variable counting the number of training steps
# Variables that affect learning rate.
cdef float num_batches_per_epoch = img_nums / batch_size
cdef int decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY) # 多少步后开始执行学习效率衰减

# Decay the learning rate exponentially based on the number of steps.
learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                global_step,
                                decay_steps,
                                LEARNING_RATE_DECAY_FACTOR,
                                staircase=True)
# 设置动态学习效率----------
''''''

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
    conv1 = maxpool2d(conv1, k=2) # 图像 30*30*32

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
init = tf.global_variables_initializer()

# if __name__ == '__main__':

# 提起pickle数据 data包含 特征+标签
data = tool_set.read_and_decode(dir_name + "train_data.pkl",img_pixel, channels)
data_1 = tool_set.read_and_decode(dir_name + "train_data_1.pkl", img_pixel, channels)

data=np.vstack((data,data_1)) #2组数据按行合并

saver = tf.train.Saver()  # 默认是保存所有变量
cdef int epoch,i

with tf.Session() as sess:
    sess.run(init)

    total_batch = int(img_nums / batch_size)

   # cdef int i 
    for epoch in range(training_epochs):

        # 每进行一个周期训练时，先将原数据按行打乱
        index = [i for i in range(len(data))]  # len(data)得到的行数
        np.random.shuffle(index)  # 将索引打乱
        data = data[index]

        avg_cost = 0.
        for i in range(total_batch):
            img, label = tool_set.next_batch(data, batch_size, img_pixel=10, channels=4)
            batch_xs = img.reshape([-1, img_size])
            batch_ys = tool_set.dense_to_one_hot(label[:, np.newaxis], label_cols)  # 生成多列标签

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys, keep_prob: dropout})
            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost:", "{:.9f}".format(avg_cost),
                  'accuracy:', sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.}))
    print("Optimization Finished!")
    save_path = saver.save(sess, dir_name+"model/save_net.ckpt")  # 保留训练的模型
    print('Saver path:', save_path)


