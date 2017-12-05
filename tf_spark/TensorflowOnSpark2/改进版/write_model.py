# -*- coding:utf-8 -*-
import tensorflow as tf

IMAGE_PIXELS=10 # 图像大小 mnist 28x28x1  (后续参考自己图像大小进行修改)
channels=3
num_class=2
learning_rate=1e-5
# Placeholders or QueueRunner/Readers for input data
x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS * channels], name="x")  # mnist 28*28*1
y_ = tf.placeholder(tf.float32, [None, num_class], name="y_")
keep=tf.placeholder(tf.float32)
x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, channels])  # mnist 数据 28x28x1 (灰度图 波段为1)


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


# Store layers weight & bias
weights = {
    # 5x5 conv, 3 input, 32 outputs 彩色图像3个输入(3个频道)，灰度图像1个输入
    'wc1': tf.Variable(tf.random_normal([5, 5, channels, 32])),  # 5X5的卷积模板
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([(1 + IMAGE_PIXELS // 4) * (1 + IMAGE_PIXELS // 4) * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_class]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_class]))
}


# 改成卷积模型
conv1 = conv2d(x_img, weights['wc1'], biases['bc1'])
conv1 = maxpool2d(conv1, k=2)
conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
conv2 = maxpool2d(conv2, k=2)
fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
fc1 = tf.nn.relu(fc1)
fc1 = tf.nn.dropout(fc1, keep)
y = tf.add(tf.matmul(fc1, weights['out']), biases['out'])


global_step = tf.Variable(0, name="global_step", trainable=False)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_op = tf.train.AdagradOptimizer(learning_rate).minimize(
    loss, global_step=global_step)

# Test trained model
label = tf.argmax(y_, 1, name="label")
prediction = tf.argmax(y, 1, name="prediction")
correct_prediction = tf.equal(prediction, label)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

