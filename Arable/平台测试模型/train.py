#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
import os
from os import path
import tensorflow as tf
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
model_fn = path.join(model_save_path, 'save_net.ckpt')

start_time = datetime.datetime.now()
print("startTime: ", start_time)

# 提起pickle数据 data包含 特征+标签
data = tool_set.read_and_decode(dir_name + "new.pkl")

isize = 10
img_channel = 3

img_pixel = isize

'''
# CNN 完整程序  训练模型
'''
# Parameters
training_epochs = 200
batch_size = 128
display_step = 10
channels = img_channel
per_process_gpu_memory_fraction = 1

# Network Parameters
img_size = isize * isize * channels  # data input (img shape: 28*28*3)
label_cols = 2  # total classes (云、非云)使用标签[1,0,0] 3维
dropout = 0.75  # Dropout, probability to keep units
img_nums = data.shape[0]

# ---------设置动态学习效率
# Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = batch_size  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

global_step = training_epochs * (img_nums // batch_size)  # Integer Variable counting the number of training steps
# Variables that affect learning rate.
num_batches_per_epoch = img_nums / batch_size
decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

# Decay the learning rate exponentially based on the number of steps.
learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                           global_step,
                                           decay_steps,
                                           LEARNING_RATE_DECAY_FACTOR,
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

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的op
init = tf.global_variables_initializer()

if __name__ == '__main__':

    saver = tf.train.Saver()  # 默认是保存所有变量
    config = tf.ConfigProto(log_device_placement=False,
                            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))
    with tf.Session(config=config) as sess:
        sess.run(init)
        # saver.restore(sess, model_fn)

        total_batch = int(img_nums / batch_size)

        for epoch in range(training_epochs):

            index = [i for i in range(len(data))]
            np.random.shuffle(index)
            data = data[index]

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
            if (epoch + 1) % display_step == 0:
                print("time: ", datetime.datetime.now())
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost:", "{:.9f}".format(avg_cost),
                      'accuracy:', sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.}))
        print("Optimization Finished!")
        save_path = saver.save(sess, model_fn)  # 保留训练的模型
        print('Saver path:', save_path)

        para = (sess.run(weights['wc1']), sess.run(weights['wc2']), sess.run(weights['wd1']), sess.run(weights['out']),
                sess.run(biases['bc1']), sess.run(biases['bc2']), sess.run(biases['bd1']), sess.run(biases['out']))
        with gzip.open(path.join(model_save_path, "para.pkl"), "wb") as f:
            pickle.dump(para, f)

        end_time = datetime.datetime.now()
        print("end_time: ", end_time)
        print("time used: ", end_time - start_time)
