#!/usr/bin/python3
# -*- coding:utf-8 -*-

"""
-----------单机版-------------------
tf读取hdfs上的数据（只支持读取.csv与.tfrecord格式数据）
hdfs dfs -put titanic_dataset.csv mnist  # 将数据上传到hdfs上

tensorflow 读取hdfs： https://github.com/fengzhongyouxia/Tensorflow/blob/master/tf_Distributed/Read_HDFS.md

tensorflow 读取本地csv数据：https://github.com/fengzhongyouxia/Tensorflow-learning/blob/master/Data%20operation/softmax%20csv.py

tensorflow 读取本地tfrecord数据：https://github.com/fengzhongyouxia/Tensorflow-learning/blob/master/Data%20operation/tfrecord%20to%20numpy.py

.csv数据
http://blog.csdn.net/wc781708249/article/details/78012990
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(os.getcwd())

#读取函数定义
def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1) # 跳过标题行
    key, value = reader.read(file_queue)
    #定义列
    defaults = [ [0], [0.], [''],[''],[0.], [0],[0],[''],[0.0]]
    #编码
    survived,pclass,name,sex,age,sibsp,parch,ticket,fare = tf.decode_csv(value, defaults)

    #处理
    gender=tf.case({tf.equal(sex,tf.constant('female')):lambda: tf.constant(1.),
                    tf.equal(sex, tf.constant('male')): lambda: tf.constant(0.),
                    }, lambda: tf.constant(-1.), exclusive=True)

    #栈
    features=tf.stack([pclass,gender,age])
    return features, survived # 返回 X，Y


def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs) # 放入在文件队列里
    example, label = read_data(file_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return example_batch, label_batch # 返回X，Y

global_step = tf.Variable(0, trainable=False)
# learning_rate = 0.1#tf.train.exponential_decay(0.1, global_step, 100, 0.0)


# Input layer
x = tf.placeholder(tf.float32, [None, 3])
y = tf.placeholder(tf.int32, [None])

# Output layer
w = tf.Variable(tf.random_normal([3, 2])) # 二分类
b = tf.Variable(tf.random_normal([2]))
# a = tf.matmul(x, w) + b
# prediction = tf.nn.softmax(a)

def inference(X):
    return tf.nn.softmax(tf.matmul(X,w)+b)

def loss(X,Y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=inference(X)))

def inputs():
    # 路径变成本地路径，就实现读取本地数据
    data_path="hdfs://xxx:8020/user/root/mnist/titanic_dataset.csv"
    x_train_batch, y_train_batch = create_pipeline(data_path, 50, num_epochs=1000)
    return x_train_batch,y_train_batch

def train(total_loss):
    learning_rate=0.1
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

def evaluate(X,Y):
    correct_prediction = tf.equal(tf.argmax(inference(X), 1), tf.cast(y, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


x_train_batch, y_train_batch =inputs()
cross_entropy=loss(x,y)
train_step=train(cross_entropy)

accuracy=evaluate(x,y)
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) #

with tf.Session() as sess:
    init.run()  #只初始化tf.global_variables_initializer() 会报错，必须还初始化tf.local_variables_initializer()
    coord = tf.train.Coordinator() #
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 线程

    try:
        print("Training: ")
        count = 0
        # curr_x_test_batch, curr_y_test_batch = sess.run([x_test, y_test])
        while not coord.should_stop() and count<2000:
            # Run training steps or whatever
            curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch]) # 必须将队列中的值取出，才能放入到feed_dict进行传递

            sess.run(train_step, feed_dict={
                x: curr_x_train_batch,
                y: curr_y_train_batch
            })

            count += 1

            ce,acc = sess.run([cross_entropy,accuracy], feed_dict={
                x: curr_x_train_batch,
                y: curr_y_train_batch
            })
            if count%100==0:
                print('Batch:',count,'loss:',ce,'accuracy:',acc)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
