#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""  单机版 从gtdata上解析数据（使用gdal读取图片数据）
# 上传 mnist_name.txt到hdfs
mnist_name.txt生成参考： 解析本地mnist图片文件名.py
zip -r mnist.zip mnist_name.txt 
hdfs dfs -mkdir mnist
hdfs dfs -put mnist.zip mnist # 上传到hdfs上的mnist文件夹下
或者 将mnist_name.txt拷至本地，在本地上读取，如果是集群 必须每台机器都的复制
执行命令：
spark-submit \
--queue default \
--num-executors 45 \
--executor-memory 2G \
--driver-memory 12G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs://xxx:8020/user/root/mnist/mnist.zip#mnist \
xxx.py 
或者
python xxx.py
"""

import tensorflow as tf
import numpy as np
import sys
try:
  from osgeo import gdal
except:
  import gdal

# 解析mnist图片在gtdata上的完整路径
file_name = []
[file_name.append('gtdata:///users/xiaoshaolin/tensorflow/mnist_test/'+i.replace('\n',''))
# for i in open('mnist/mnist_name.txt')] # 从hdfs上读取文件mnist_name.txt
for i in open('mnist_name.txt')] # 从本地读取文件mnist_name.txt

# 删除最后一个字符
def DelLastChar(str):
    str_list=list(str)
    str_list.pop()
    return "".join(str_list)

'''
# 多波段图像(遥感图像)提取每个波段信息转换成数组（波段数>=4 或者 波段数<=3）
# 一般的方法如：opencv，PIL，skimage 最多只能读取3个波段
# path 图像的路径
# return： 图像数组
'''
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

np.random.shuffle(file_name)
# 从gtdata上解析数据
for i,img_path in enumerate(file_name[:1000]):
   img_path=DelLastChar(img_path)
   labels=img_path.split('/')[-2]  # 解析标签 （单列标签 非one_hot标签）
   imgs=Multiband2Array(img_path) # 解析图像数据
   data1=imgs.flatten() # 按行展成一行  或 data1=img.reshape((-1,28*28*1))
   data2 = np.append(data1, labels)[np.newaxis, :]  # 数据+标签

   if i == 0:
      data = data2
   else:
      data = np.vstack((data, data2))  # 上下合并

print(data.shape)
# exit(-1)
# 打乱数据
np.random.shuffle(data)

# 选取0.3测试数据与0.7训练数据
train_data=data[:int(len(data)*0.7)]
test_data=data[int(len(data)*0.7):]

# -------------数据解析完成----------------------------------------#

start_index=0
def next_batch(data,batch_size):
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
    img = data1[:, 0:-1]
    img = img.astype(float)

    label = data1[:, -1:]
    label = label.astype(float)  # 类型转换

    return img,label


def main(_):
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    # y=tf.nn.softmax(tf.nn.relu(y))
    y = tf.nn.softmax(y)
    # Define loss and optimizer
    # y_ = tf.placeholder(tf.float32, [None, 10])
    y_ = tf.placeholder(tf.float32, [None, ]) # 使用非one_hot标签
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.to_int64(y_)))  # 使用非one_hot标签

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = tf.train.AdagradOptimizer(0.5).minimize(
        cross_entropy, global_step=global_step)

    # Test trained model
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.to_int64(y_))  # 使用非one_hot标签

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess=tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Loop until the supervisor shuts down (or 2000 steps have completed).
    step = 0
    while step < 1000:
        batch_xs, batch_ys = next_batch(train_data,100)
        _, step = sess.run([train_op, global_step], feed_dict={x: batch_xs, y_: batch_ys.flatten()})
        # print("Step %d in task %d" % (step, FLAGS.task_index))
        if step % 100 == 0:
            print("accuracy: %f" % sess.run(accuracy, feed_dict={x: test_data[:, :-1],
                                                                 y_: test_data[:, -1]}))

if __name__ == "__main__":
    tf.app.run()
