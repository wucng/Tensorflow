#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import os
from os import path

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

import numpy as np
import tool_set
import datetime
import pickle
import gzip

# 训练集文件路径
dir_name = r''
model_save_path = r"model/"
if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)  # 创建输出文件目录
model_fn = path.join(model_save_path, 'save_net.ckpt')  # 存放掩膜影像

start_time = datetime.datetime.now()
print("startTime: ", start_time)

# 提起pickle数据 data包含 特征+标签
data = tool_set.read_and_decode(dir_name + "04_pool_50p.pkl")

isize = 2
img_channel = 4

img_pixel = isize

'''
# CNN 完整程序  训练模型
'''
# Parameters
training_epochs = 50
batch_size = 128
display_step = 10
channels = img_channel

# Network Parameters
img_size = isize * isize * channels  # data input (img shape: 28*28*3)
label_cols = 2  # total classes (云、非云)使用标签[1,0,0] 3维
dropout = 0.75  # Dropout, probability to keep units
img_nums = data.shape[0]

# ---------设置动态学习效率
# Constants describing the training process.
batch = tf.Variable(0)
decay_steps = img_nums

# Decay the learning rate exponentially based on the number of steps.
learning_rate = tf.train.exponential_decay(0.01,
                                           batch * batch_size,
                                           decay_steps,
                                           0.95,
                                           staircase=True)
# 设置动态学习效率----------
''''''

# with tf.device('/gpu:0'):
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
    x = tf.reshape(x, shape=[-1, img_pixel, img_pixel, channels])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # FCN
    fcn1 = tf.nn.conv2d(conv1, weights['wcn1'], strides=[1, 1, 1, 1], padding='VALID')
    fcn1 = tf.nn.bias_add(fcn1, biases['bcn1'])
    fcn1 = tf.nn.relu(fcn1)
    # fcn1 = tf.nn.dropout(fcn1, dropout)

    # FCN
    fcn2 = tf.nn.conv2d(fcn1, weights['wcn2'], strides=[1, 1, 1, 1], padding='VALID')
    fcn2 = tf.nn.bias_add(fcn2, biases['bcn2'])
    fcn2 = tf.nn.dropout(fcn2, dropout)

    # out = fcn2
    out = tf.reshape(fcn2, shape=[-1, 2])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 3 input, 32 outputs 彩色图像3个输入(3个频道)，灰度图像1个输入
    'wc1': tf.Variable(tf.random_normal([2, 2, channels, 64])),  # 5X5的卷积模板
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wcn1': tf.Variable(tf.random_normal([1, 1, 64, 4096])),
    # 1024 inputs, 10 outputs (class prediction)
    'wcn2': tf.Variable(tf.random_normal([1, 1, 4096, label_cols]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bcn1': tf.Variable(tf.random_normal([4096])),
    'bcn2': tf.Variable(tf.random_normal([label_cols]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)
soft_pred = tf.nn.softmax(pred, 1)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=batch)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的op
init = tf.global_variables_initializer()

if __name__ == '__main__':

    saver = tf.train.Saver()  # 默认是保存所有变量
    config = tf.ConfigProto(log_device_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(init)
        # saver.restore(sess, model_fn)

        total_batch = int(img_nums / batch_size)

        for epoch in range(training_epochs):

            np.random.shuffle(data)

            avg_cost = 0.
            for i in range(total_batch):
                img, label = tool_set.next_batch(data, batch_size, img_pixel=isize, channels=img_channel)
                batch_xs = img.reshape([-1, img_size])
                batch_ys = tool_set.dense_to_one_hot(label[:, np.newaxis], label_cols)  # 生成多列标签

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_xs,
                                                                       y: batch_ys, keep_prob: dropout})
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            if epoch < 11 or (epoch + 1) % display_step == 0:
                l, lr, accur = sess.run([cost, learning_rate, accuracy],
                                        feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                print("time: ", datetime.datetime.now())
                print("Epoch:", '%04d' % (epoch + 1),
                      "learning rate:", "{:.5f}".format(lr),
                      "cost:", "{:.9f}".format(l),
                      'accuracy:', accur)

        # Export model
        print('Exporting trained model to %s' % model_save_path)
        if tf.gfile.Exists(model_save_path):
            tf.gfile.DeleteRecursively(model_save_path)
        builder = saved_model_builder.SavedModelBuilder(model_save_path)

        # Build the signature_def_map.
        tensor_info_x = utils.build_tensor_info(x)
        tensor_info_y = utils.build_tensor_info(soft_pred)
        tensor_info_keep_prob = utils.build_tensor_info(keep_prob)

        prediction_signature = signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x,
                    'keep_prob': tensor_info_keep_prob},
            outputs={'scores': tensor_info_y},
            method_name=signature_constants.PREDICT_METHOD_NAME)

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map={'predict_images': prediction_signature}, legacy_init_op=legacy_init_op)

        builder.save()

        print('Done exporting!')

        end_time = datetime.datetime.now()
        print("end_time: ", end_time)
        print("time used: ", end_time - start_time)
