# 1、训练
```python
#!/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.framework import ops

# ops.reset_default_graph()
"""
cnn
"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义一些参数
batch_size = 128
droup_out = 0.7
learn_rate = 0.1
num_steps = 10
disp_step = 2

with tf.Graph().as_default() as graph:
    # mnist图像大小是28x28 分成0~9 共10类
    x=tf.placeholder(tf.float32,[None,28*28*1])
    y_=tf.placeholder(tf.float32,[None,10])
    keep=tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, name='MODE')

    x_img=tf.reshape(x,[-1,28,28,1])

    def batch_norm_layer(inputT, is_training=True, scope=None):
        # Note: is_training is tf.placeholder(tf.bool) type
        return tf.cond(is_training,
                       lambda: batch_norm(inputT, is_training=True,
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                       lambda: batch_norm(inputT, is_training=False,
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                                          scope=scope, reuse = True))


    def conv2d(input, kernel_size, input_size, output_size, is_training, name):
        with tf.name_scope(name) as scope:
            with tf.variable_scope(name):
                # scope.reuse_variables()
                w = tf.get_variable('w', [kernel_size, kernel_size, input_size, output_size], tf.float32,
                                    initializer=tf.random_uniform_initializer) * 0.001
                b = tf.get_variable('b', [output_size], tf.float32, initializer=tf.random_normal_initializer) + 0.1
                conv = tf.nn.conv2d(input, w, [1, 1, 1, 1], padding="SAME")
                conv = tf.nn.bias_add(conv, b)
                conv = batch_norm_layer(conv, is_training, scope)
                conv = tf.nn.relu(conv)
        return conv

    def fc_layer(input,input_size,output_size,is_training,name):
        with tf.name_scope(name) as scope:
            with tf.variable_scope(name):
                w = tf.get_variable('w', [input_size, output_size], tf.float32,
                                    initializer=tf.random_uniform_initializer) * 0.001
                b = tf.get_variable('b', [output_size], tf.float32, initializer=tf.random_normal_initializer) + 0.1
                fc=tf.nn.bias_add(tf.matmul(input,w),b)
                fc=batch_norm_layer(fc,is_training,scope)
                # fc = tf.nn.relu(fc)
                return fc

    # convolution1
    conv1 = conv2d(tf.image.convert_image_dtype(x_img, tf.float32),
                   3, 1, 32, is_training, 'conv1')
    conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")  # [n,14,14,32]
    conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    conv1 = tf.nn.dropout(conv1, keep)

    # convolution2
    conv2 = conv2d(conv1,
                   3, 32, 64, is_training, 'conv2')
    conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")  # [n,7,7,64]
    conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    conv2 = tf.nn.dropout(conv2, keep)

    # full connect
    fc1=tf.reshape(conv2,[-1,7*7*64])

    fc1=fc_layer(fc1,7*7*64,512,is_training,'fc1')
    fc1=tf.nn.relu(fc1)
    fc1=tf.nn.dropout(fc1,keep)

    fc2=fc_layer(fc1,512,10,is_training,'output')
    y=tf.nn.softmax(fc2)

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

    train_op=tf.train.AdamOptimizer(learn_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.InteractiveSession(graph=graph)

saver=tf.train.Saver()

tf.global_variables_initializer().run()

# 验证之前是否已经保存了检查点文件
ckpt = tf.train.get_checkpoint_state('./output/')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

for step in range(num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    train_op.run({x:batch_xs,y_:batch_ys,keep:droup_out,is_training:True})
    if step % disp_step==0:
        print("step",step,'acc',accuracy.eval({x:batch_xs,y_:batch_ys,keep:droup_out,is_training:True}),
              'loss',loss.eval({x:batch_xs,y_:batch_ys,keep:droup_out,is_training:True}))

saver.save(sess, './output/model.ckpt')

# test acc
print('test acc',accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep:1.,is_training:False}))

sess.close()
```
# 2、查看变量名
```python
import tensorflow as tf
import os

logdir='./output/'

from tensorflow.python import pywrap_tensorflow
# checkpoint_path = os.path.join(model_dir, "model.ckpt-9999")
ckpt = tf.train.get_checkpoint_state(logdir)
reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key))
```

# 3、不加载最后一层的所有变量（随机初始化）
```
#!/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import slim
# from tensorflow.python.framework import ops

# ops.reset_default_graph()
"""
cnn
"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义一些参数
batch_size = 128
droup_out = 0.7
learn_rate = 0.1
num_steps = 10
disp_step = 2

with tf.Graph().as_default() as graph:
    # mnist图像大小是28x28 分成0~9 共10类
    x=tf.placeholder(tf.float32,[None,28*28*1])
    y_=tf.placeholder(tf.float32,[None,10])
    keep=tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, name='MODE')

    x_img=tf.reshape(x,[-1,28,28,1])

    def batch_norm_layer(inputT, is_training=True, scope=None):
        # Note: is_training is tf.placeholder(tf.bool) type
        return tf.cond(is_training,
                       lambda: batch_norm(inputT, is_training=True,
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                       lambda: batch_norm(inputT, is_training=False,
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                                          scope=scope, reuse = True))


    def conv2d(input, kernel_size, input_size, output_size, is_training, name):
        with tf.name_scope(name) as scope:
            with tf.variable_scope(name):
                # scope.reuse_variables()
                w = tf.get_variable('w', [kernel_size, kernel_size, input_size, output_size], tf.float32,
                                    initializer=tf.random_uniform_initializer) * 0.001
                b = tf.get_variable('b', [output_size], tf.float32, initializer=tf.random_normal_initializer) + 0.1
                conv = tf.nn.conv2d(input, w, [1, 1, 1, 1], padding="SAME")
                conv = tf.nn.bias_add(conv, b)
                conv = batch_norm_layer(conv, is_training, scope)
                conv = tf.nn.relu(conv)
        return conv

    def fc_layer(input,input_size,output_size,is_training,name):
        with tf.name_scope(name) as scope:
            with tf.variable_scope(name):
                w = tf.get_variable('w', [input_size, output_size], tf.float32,
                                    initializer=tf.random_uniform_initializer) * 0.001
                b = tf.get_variable('b', [output_size], tf.float32, initializer=tf.random_normal_initializer) + 0.1
                fc=tf.nn.bias_add(tf.matmul(input,w),b)
                fc=batch_norm_layer(fc,is_training,scope)
                # fc = tf.nn.relu(fc)
                return fc

    # convolution1
    conv1 = conv2d(tf.image.convert_image_dtype(x_img, tf.float32),
                   3, 1, 32, is_training, 'conv1')
    conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")  # [n,14,14,32]
    conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    conv1 = tf.nn.dropout(conv1, keep)

    # convolution2
    conv2 = conv2d(conv1,
                   3, 32, 64, is_training, 'conv2')
    conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")  # [n,7,7,64]
    conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    conv2 = tf.nn.dropout(conv2, keep)

    # full connect
    fc1=tf.reshape(conv2,[-1,7*7*64])

    fc1=fc_layer(fc1,7*7*64,512,is_training,'fc1')
    fc1=tf.nn.relu(fc1)
    fc1=tf.nn.dropout(fc1,keep)

    fc2=fc_layer(fc1,512,10,is_training,'output')
    y=tf.nn.softmax(fc2)

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

    train_op=tf.train.AdamOptimizer(learn_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.InteractiveSession(graph=graph)

# saver=tf.train.Saver() # 默认加载所有变量
saver=tf.train.Saver(slim.get_variables_to_restore(exclude=['output'])) # 排除output相关的所有变量

tf.global_variables_initializer().run()

# 验证之前是否已经保存了检查点文件
ckpt = tf.train.get_checkpoint_state('./output/')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

for step in range(num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    train_op.run({x:batch_xs,y_:batch_ys,keep:droup_out,is_training:True})
    if step % disp_step==0:
        print("step",step,'acc',accuracy.eval({x:batch_xs,y_:batch_ys,keep:droup_out,is_training:True}),
              'loss',loss.eval({x:batch_xs,y_:batch_ys,keep:droup_out,is_training:True}))

# saver.save(sess, './output/model.ckpt')

# test acc
print('test acc',accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep:1.,is_training:False}))

sess.close()
```
