# -*- coding: utf-8 -*-

from __future__ import print_function
from osgeo import gdal, ogr,osr
from osgeo.gdalconst import *
import numpy as np
import os
from os import path
#import numexpr as ne
import tensorflow as tf
from numba.decorators import jit

img_path="/home/wu/Water_extract/data/shp2/1231.tif"  # 输入影像
vector_fn_sample="/home/wu/Water_extract/data/shp2/1234.shp" # 输入shp文件
raster_fn_sample =img_path+"sample_mask.tiff"  # 存放掩膜影像



if not os.path.exists(raster_fn_sample): #获取掩膜影像

    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8","NO")

    gdal.AllRegister() #注册驱动
    ogr.RegisterAll()


    # img_path=input('输入影像路径:')
    srcDS=gdal.Open(img_path,GA_ReadOnly)# 只读方式打开影像
    geoTrans = srcDS.GetGeoTransform() # 获取地理参考6参数
    srcPro=srcDS.GetProjection() # 获取坐标引用

    # prjRef
    # prjRef = osr.SpatialReference(srcPro) # (栅格坐标参考)转换成矢量坐标参考

    srcXSize=srcDS.RasterXSize # 宽度
    srcYSize=srcDS.RasterYSize # 高度
    nbands=srcDS.RasterCount # 波段数

    module_path = path.dirname(__file__) # 返回脚本文件所在的工作目录

    # 生成掩膜影像(确定样本提取)
    # vector_fn_sample=input('输入掩膜shp路径:')
    # vector_fn_sample=r"C:\Users\Administrator\Desktop\shp2\1234.shp" # 输入shp文件


    if os.path.exists(raster_fn_sample):
        gdal.GetDriverByName('GTiff').Delete(raster_fn_sample)# 删除掉样本提取掩膜

    source_ds_1 = ogr.Open(vector_fn_sample) # 打开矢量文件
    source_layer_1 = source_ds_1.GetLayer() # 获取图层  （包含图层中所有特征  所有几何体）
    mark_ds = gdal.GetDriverByName('GTiff').Create(raster_fn_sample, srcXSize, srcYSize, 1, gdal.GDT_Byte)  # 1表示1个波段，按原始影像大小生成 掩膜
    mark_ds.SetGeoTransform(geoTrans)  # 设置掩膜的地理参考
    mark_ds.SetProjection(srcPro)  # 设置掩膜坐标引用
    band = mark_ds.GetRasterBand(1)  # 获取第一个波段(影像只有一个波段)
    band.SetNoDataValue(0)  # 将这个波段的值全设置为0
    # Rasterize 矢量栅格化
    gdal.RasterizeLayer(mark_ds, [1], source_layer_1, burn_values=[1])  # 几何体内的值全设置为1
    mark_ds.FlushCache()  # 将数据写入文件
    source_ds_1=None
    srcDS=None
    # markDS_sample=gdal.Open(raster_fn_sample,GA_ReadOnly)# 只读方式打开掩膜影像


srcDS=gdal.Open(img_path,GA_ReadOnly)# 只读方式打开影像
srcXSize=srcDS.RasterXSize # 宽度
srcYSize=srcDS.RasterYSize # 高度
nbands=srcDS.RasterCount # 波段数
# 一次性读取图像所有数据（image）


for band in range(nbands):
    # band=int(band)
    band += 1
    # print "[ GETTING BAND ]: ", band
    srcband = srcDS.GetRasterBand(band)  # 获取该波段
    if srcband is None:
        continue

    # Read raster as arrays 类似RasterIO（C++）
    dataraster = srcband.ReadAsArray(0, 0, srcXSize, srcYSize, srcXSize, srcYSize).astype(np.uint8)  # 像素值转到[0~255]

    if band == 1:
        data = dataraster.reshape((srcYSize, srcXSize, 1))
    else:
        # 将每个波段的数组很并到一个3s维数组中
        data = np.append(data, dataraster.reshape((srcYSize, srcXSize, 1)), axis=2)
    del dataraster, srcband

# 全部进行归一化操作
#ne.set_num_threads(4)
#ne.set_vml_num_threads(2)
#data = ne.evaluate('data * (1. / 255) - 0.5')  # 数据归一化到 -0.5～0.5
data = data * (1. / 255) - 0.5 # 数据归一化到 -0.5～0.5

markDS_sample=gdal.Open(raster_fn_sample,GA_ReadOnly)# 只读方式打开掩膜影像
data_label = markDS_sample.ReadAsArray(0, 0, srcXSize, srcYSize).astype(np.float32)

del srcDS
del markDS_sample


#-------------------------------------------------------

'''
# CNN 完整程序  训练模型
'''
# Parameters
cdef float learning_rate = 10**(-3)
cdef int training_epochs = 500
# training_iters = 200000
cdef int batch_size = 128
cdef int display_step = 5
cdef int img_pixel=100
cdef int channels=4

# Network Parameters
cdef int img_size = img_pixel*img_pixel*channels # data input (img shape: 28*28*3)
# label_cols = 2 # total classes (云、非云)使用标签[1,0,0] 3维
cdef int label_cols = img_pixel*img_pixel
cdef float dropout = 0.75 # Dropout, probability to keep units
cdef int img_nums=28874

cdef int total_batch = srcYSize//img_pixel*srcXSize//img_pixel*2

'''
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
''''''
'''

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

    conv3=conv2d(conv2,weights['wc3'],biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)  # 图像 15*15*64

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    conv4 = maxpool2d(conv4, k=2)  # 图像 15*15*64

    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
    conv5 = maxpool2d(conv5, k=2)  # 图像 15*15*64


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]]) # [None,15*15*64]
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

    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),

    'wc4': tf.Variable(tf.random_normal([5, 5, 128, 256])),

    'wc5': tf.Variable(tf.random_normal([5, 5, 256, 256])),

    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, label_cols]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([256])),

    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([label_cols]))
}


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
cost=tf.reduce_mean(tf.reduce_sum(tf.square(y-pred),reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
accuracy=tf.reduce_mean(tf.reduce_sum(tf.square(y-pred),reduction_indices=[1]))

# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#初始化所有的op
init = tf.global_variables_initializer()

# if __name__ == '__main__':

# 提起pickle数据 data包含 特征+标签
# data = tool_set.read_and_decode(dir_name + "train_data.pkl",img_pixel, channels)
# data_1 = tool_set.read_and_decode(dir_name_1 + "train_data.pkl", img_pixel, channels)

# data=np.vstack((data,data_1)) #2组数据按行合并

saver = tf.train.Saver()  # 默认是保存所有变量

cdef int epoch,i,j
with tf.Session() as sess:
    sess.run(init)

    # total_batch = srcYSize//img_pixel*srcXSize//img_pixel*3

    def compute():
        for epoch in range(training_epochs):

            avg_cost = 0.
            for i in range(total_batch):
                flag=True
                for j in range(batch_size):
                    yi = np.random.random_integers(0, srcYSize-img_pixel)
                    xi = np.random.random_integers(0, srcXSize-img_pixel)
                    img=data[yi:yi+img_pixel,xi:xi+img_pixel]
                    img = img.reshape([-1, img_size])
                    label = data_label[yi:yi + img_pixel, xi:xi + img_pixel]
                    label = label.reshape([-1, label_cols])
                    if flag:
                        batch_xs=img
                        batch_ys=label
                        flag=False
                    else:
                        batch_xs=np.vstack((batch_xs,img))
                        batch_ys=np.vstack((batch_ys,label))


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


    try:
        jit(compute(), target='cuda')
    except:
        pass

    print("Optimization Finished!")
    save_path = saver.save(sess, "/home/wu/Water_extract/data/shp2/model/save_net.ckpt")  # 保留训练的模型
    print('Saver path:', save_path)

