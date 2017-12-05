# -*- coding:utf-8 -*-

"""
单机测试模型精度（舍弃全连接层），对应mnist_dist.py
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


IMAGE_PIXELS=10 # 图像大小 mnist 28x28x1  (后续参考自己图像大小进行修改)
channels=3
num_class=2
dropout = 0.5
learning_rate=1e-6
batch_size=200
train=0

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
    #mean = np.reshape(np.average(img, 1), [np.shape(img)[0], 1])
    #std = np.reshape(np.std(img, 1), [np.shape(img)[0], 1])

    #img = (img - mean) / std
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

def maxpool2d2(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')  # strides中间两个为2 表示x,y方向都间隔1个取样

# Store layers weight & bias
weights = {
    # 5x5 conv, 3 input, 32 outputs 彩色图像3个输入(3个频道)，灰度图像1个输入
    'wc1': tf.get_variable('wc1',[3,3,channels,64],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),  # 5X5的卷积模板

    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.get_variable('wc2',[3,3,64,128],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
    # 'wc3': tf.Variable(tf.random_normal([3, 3, 256, 128])),
    'wc4': tf.get_variable('wc4',[3,3,128,num_class],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
    # fully connected, 7*7*64 inputs, 1024 outputs
    # 'wd1': tf.Variable(tf.random_normal([(1+IMAGE_PIXELS // 4) * (1+IMAGE_PIXELS // 4) * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    # 'out': tf.Variable(tf.random_normal([1024, num_class]))
}

biases = {
    'bc1': tf.get_variable('bc1',[64],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
    'bc2': tf.get_variable('bc2',[128],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
    # 'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.get_variable('bc4',[num_class],dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
    # 'bd1': tf.Variable(tf.random_normal([1024])),
    # 'out': tf.Variable(tf.random_normal([num_class]))
}

# Placeholders or QueueRunner/Readers for input data
x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS * channels], name="x")  # mnist 28*28*1
y_ = tf.placeholder(tf.float32, [None, num_class], name="y_")
keep=tf.placeholder(tf.float32)

x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, channels])  # mnist 数据 28x28x1 (灰度图 波段为1)
# tf.summary.image("x_img", x_img)

# 改成卷积模型
conv1 = conv2d(x_img, weights['wc1'], biases['bc1'])
conv1 = maxpool2d(conv1, k=2)
# conv1 = tf.nn.dropout(conv1, keep)
conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
conv2 = maxpool2d(conv2, k=2)
conv2 = tf.nn.dropout(conv2, keep)
# conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
# conv3 = tf.nn.dropout(conv3, keep)
conv4 = conv2d(conv2, weights['wc4'], biases['bc4'])
conv4 = maxpool2d2(conv4, k=2)
y = tf.reshape(conv4, [-1, num_class])

# fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
# fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
# fc1 = tf.nn.relu(fc1)
# if args.mode == "train" or args.mode == "retrain":
#   fc1 = tf.nn.dropout(fc1, dropout)
# y = tf.add(tf.matmul(fc1, weights['out']), biases['out'])


# global_step = tf.Variable(0)

global_step = tf.Variable(0, name="global_step", trainable=False)

# loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# tf.summary.scalar("loss", loss)
# tf.train.GradientDescentOptimizer()
# train_op = tf.train.AdagradOptimizer(learning_rate).minimize(
#     loss, global_step=global_step)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
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


data=read_and_decode_1('../13.pkl')
np.random.shuffle(data) # 随机打乱


with tf.Session() as sess:
    if train:
        if os.path.exists('model_aralbe/checkpoint'):
            saver.restore(sess, 'model_aralbe/model.ckpt-9999')
        else:
            sess.run(init_op)
        for step in range(10000):
            if not step%500:np.random.shuffle(data) # 随机打乱
            batch_xs, batch_ys = next_batch(data, batch_size, img_pixel=10, channels=3)
            batch_ys = dense_to_one_hot2(batch_ys, 2)
            feed = {x: batch_xs, y_: batch_ys,keep:dropout}
            sess.run(train_op,feed_dict=feed)
            acc = sess.run(accuracy, {x: batch_xs, y_: batch_ys, keep: 1.})
            if acc > 0.9:
                saver.save(sess, 'model_aralbe/model.ckpt', global_step=step)
            if step%100==0:
                [loss1]=sess.run([loss],feed)
                print("step",step,"loss",loss1,"acc",acc)
        saver.save(sess, 'model_aralbe/model.ckpt')

    else:
        saver.restore(sess,'model_aralbe/model.ckpt-9999')
        batch_xs, batch_ys = next_batch(data,20000,img_pixel=10,channels=3)
        batch_ys=dense_to_one_hot2(batch_ys,2)
        feed = {x: batch_xs, y_: batch_ys,keep:1.}
        acc = sess.run(accuracy, feed_dict=feed)
        print("acc: {0}".format(acc))

