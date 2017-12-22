#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
传入一张影像进行预测，并提取水体范围
"""

from __future__ import print_function
from os import path
import tensorflow as tf
import numpy as np
import datetime
from osgeo import gdal, ogr
import tool_set
import cv2
import sys

#映射字典，2017.10.18,by xjxf __start
bid_dict={
    0:0,
    1:15,
    2:16,
    3:17,
    4:21,
    5:30,
    6:41,
    7:43,
    8:46,
    9:50,
    10:54,
    11:56,
    12:65,
    13:90
}
#映射字典，2017.10.18,by xjxf __end
starttime = datetime.datetime.now()
print(starttime)
# 训练集文件路径
step =1

dir_name = 'train_data/13.tif'  # 影像路径
dir_name_mask='train_data/13_mask.tif' #掩膜路径
dir_output = ''

model_path = "model_03/save_net.ckpt"

# # 输出文件路径设置
fpa_path = path.join(dir_output, 'classification_output.txt')
fpa = open(fpa_path, "a")
# fpa.close()

# time格式化
print("startTime: ", starttime)

'''
# CNN 完整程序  测试模型
'''
# Parameters
# learning_rate = 10**(-5)
training_epochs = 50
# training_iters = 200000
batch_size = 1
display_step = 10
isize=400
img_pixel_h=isize
img_pixel_w=isize
channels =3

# Network Parameters
img_size = isize * isize * channels  # data input (img shape: 28*28*3)
label_cols =3 # total classes (云、非云)使用标签[1,0,0] 3维
dropout = 0.75  # Dropout, probability to keep units
img_nums = 408408

# ---------设置动态学习效率
# ---------设置动态学习效率
# Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = batch_size  # Epochs after which learning rate decays.        #问题3，不知道干什么用的    by xjxf
LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.                      #问题4，不知道干什么用的   by xjxf
INITIAL_LEARNING_RATE = 0.00015 # Initial learning rate.                                 #问题5，不知道干什么用的   by xjxf

global_step =tf.Variable(0)  # Integer Variable counting the number of training steps     # //是整数除法
# global_step = training_epochs * (img_nums // batch_size)
# print("global_step:",global_step)
# Variables that affect learning rate.
num_batches_per_epoch = img_nums / batch_size
# num_batches_per_epoch = int(img_nums / batch_size)*2
# decay_steps=3
# decay_steps = int((num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)/2)
decay_steps = int(num_batches_per_epoch*50 )
print("decay_steps:",decay_steps)
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



layer_dict={
    'wc1_1':[3,3,channels,64],
    'wc1_2':[3,3,64,64],

    'wc2_1':[3,3,64,128],
    'wc2_2':[3,3,128,128],

    'wc3_1':[3,3,128,256],
    'wc3_2':[3,3,256,256],
    'wc3_3':[3,3,256,256],

    'wc4_1':[3,3,256,512],
    'wc4_2':[3,3,512,512],
    'wc4_3':[3,3,512,512],

    'wc5_1':[3,3,512,512],
    'wc5_2':[3,3,512,512],
    'wc5_3':[3,3,512,512],

    'wf_6':[7,7,512,4096],
    'wf_7':[1,1,4096,4096],

    'w_out':[1,1,4096,label_cols],

    'up_sample_p_4':[1,1,512,label_cols],

    'up_sample_p_3':[1,1,256,label_cols],

    'up_sample_p_2':[1,1,128,label_cols]

}
#定义一个简化卷积层操作的函数  2017.09.29 by xhxf__start
def conv_relu(bottom, name, stride=1):
    # bottom_new=np.lib.pad(bottom,(pad),"contant",constant_values=0)
    layer_name=name
    layer_pram=layer_dict[name]
    with tf.variable_scope(layer_name):
        with tf.variable_scope('weights'):
            weights_1=tf.get_variable(name+'_kernel',shape=layer_pram,initializer=tf.glorot_uniform_initializer())
            tf.summary.histogram(layer_name+'/weights',weights_1)  #可视化观察变量  by xjxf, 2017.10.11
        with tf.variable_scope('biases'):
            biases_1=tf.get_variable(name+'biases',shape=layer_pram[3],initializer=tf.glorot_uniform_initializer())
            tf.summary.histogram(layer_name+'/biasses',biases_1)   #可视化观察变量  by xjxf, 2017.10.11


    conv=tf.nn.conv2d(bottom,weights_1,strides=[1,stride,stride,1],padding="SAME")
    conv=tf.nn.bias_add(conv,biases_1)
    return conv, tf.nn.relu(conv)
#定义一个简化卷积层操作的函数  2017.09.29 by xhxf__ned


#定义一个简化池化层操作过程的函数  2017.09.30 by xjjxf  _start
def max_pool(bottom,k=2):
    return tf.nn.max_pool(bottom,ksize=[1,k,k,1],strides=[1,k,k,1],padding="SAME")
    # return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)
#定义一个简化池化层操作过程的函数  2017.09.30 by xjjxf  _start



# Create model
def conv_net(x, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, img_pixel_h, img_pixel_w, channels])  # 彩色图像3个频道，如果是灰度图就是1

    # Convolution Layer
    conv1_1,relu1_1= conv_relu(x, 'wc1_1')  # 图像60*60*32
    conv1_2,relu1_2=conv_relu(relu1_1,'wc1_2')
    pool_1=max_pool(relu1_2)

    conv2_1, relu2_1 = conv_relu(pool_1, 'wc2_1')  # 图像60*60*32
    conv2_2, relu2_2 = conv_relu(relu2_1, 'wc2_2')
    pool_2 = max_pool(relu2_2)

    conv3_1, relu3_1 = conv_relu(pool_2, 'wc3_1')  # 图像60*60*32
    conv3_2, relu3_2 = conv_relu(relu3_1, 'wc3_2')
    conv3_3, relu3_3 = conv_relu(relu3_2, 'wc3_3')
    pool_3 = max_pool(relu3_3)

    conv4_1, relu4_1 = conv_relu(pool_3,'wc4_1')  # 图像60*60*32
    conv4_2, relu4_2 = conv_relu(relu4_1, 'wc4_2')
    conv4_3, relu4_3 = conv_relu(relu4_2, 'wc4_3')
    pool_4 = max_pool(relu4_3)

    conv5_1, relu5_1 = conv_relu(pool_4, 'wc5_1')  # 图像60*60*32
    conv5_2, relu5_2 = conv_relu(relu5_1, 'wc5_2')
    conv5_3, relu5_3 = conv_relu(relu5_2, 'wc5_3')
    pool_5 = max_pool(relu5_3)

    fc_6,relu_fc6 = conv_relu(pool_5,'wf_6')
    relu_fc6=tf.nn.dropout(relu_fc6,dropout)

    fc_7,relu_fc7=conv_relu(relu_fc6,'wf_7')
    relu_fc7=tf.nn.dropout(relu_fc7,dropout)

    out_1,relu_out1=conv_relu(relu_fc7,'w_out')


    # fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])  # [None,15*15*64]
    # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # fc1 = tf.nn.relu(fc1)
    # # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)
    #
    # # Output, class prediction
    # out_1 = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    #上采样至pool_4的大小  2017.10.13,by xjxf   __start
    out_end_1=tf.nn.conv2d_transpose(out_1,tf.get_variable("up_sample_1",shape=[4,4,label_cols,label_cols],
                                                       initializer= tf.glorot_uniform_initializer( )),
                                 output_shape=[batch_size,img_pixel_h//16,img_pixel_w//16,label_cols],
                                 strides=[1,2,2,1],padding="SAME")
    # 上采样至pool_4的大小  2017.10.13,by xjxf   __end

    #让pool_4卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start
    upsample_pool4,upsample_pool4_relu=conv_relu(pool_4,'up_sample_p_4')
    out_upsample_pool4=upsample_pool4+out_end_1
    out_end_2=tf.nn.conv2d_transpose(out_upsample_pool4,tf.get_variable("up_sample_4",shape=[4,4,label_cols,label_cols],
                                                                        initializer=tf.glorot_uniform_initializer()),
                                     output_shape=[batch_size,img_pixel_h//8,img_pixel_w//8,label_cols],
                                     strides=[1,2,2,1],padding='SAME')
    # 让pool_4卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start

    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start
    upsample_pool3, upsample_pool3_relu = conv_relu(pool_3, 'up_sample_p_3')
    out_upsample_pool3 = upsample_pool3 + out_end_2
    out_end_3 = tf.nn.conv2d_transpose(out_upsample_pool3,
                                       tf.get_variable("up_sample_3", shape=[4, 4, label_cols, label_cols],
                                                       initializer=tf.glorot_uniform_initializer()),
                                       output_shape=[batch_size, img_pixel_h//4, img_pixel_w//4, label_cols],
                                       strides=[1, 2, 2, 1], padding='SAME')
    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start

    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.16,by xjxf  __start
    upsample_pool2, upsample_pool2_relu = conv_relu(pool_2, 'up_sample_p_2')
    out_upsample_pool2 = upsample_pool2 + out_end_3
    out_end_4 = tf.nn.conv2d_transpose(out_upsample_pool2,
                                       tf.get_variable("up_sample_2", shape=[8, 8, label_cols, label_cols],
                                                       initializer=tf.glorot_uniform_initializer()),
                                       output_shape=[batch_size, img_pixel_h, img_pixel_w, label_cols],
                                       strides=[1, 4, 4, 1], padding='SAME')
    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.16,by xjxf  __start
    return out_end_4





# Construct model
pred = conv_net(x, keep_prob)
pred_new=tf.reshape(pred,[-1,label_cols])
# Define loss and optimizer



#可视化cost值,by xjxf 2017.10.11 __start
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred_new))
    tf.summary.scalar('cost',cost)
#可视化cost值,by xjxf 2017.10.11 __end
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred_new, 1), tf.argmax(y, 1))
#可视化cost值,by xjxf 2017.10.11 __start
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy',accuracy)
#可视化cost值,by xjxf 2017.10.11 __end



def polypro(argv):

    if len(argv) >1:
        dir_name = 'PL_img/train_img_'+argv[1].zfill(2)+'.tif' # 影像路径
        dir_name_mask = 'PL_img/train_img_mask_'+argv[1].zfill(2) +'.tif' # 掩膜路径
    else:
        dir_name = 'train_data/21.tif'  # 影像路径
        dir_name_mask = 'train_data/21_mask.tif'  # 掩膜路径

    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")

    srcDS = gdal.Open(dir_name, gdal.GA_ReadOnly)  # 只读方式打开原始影像
    maskDS=gdal.Open(dir_name_mask,gdal.GA_ReadOnly) #只读方式打开原始影像
    maskXSize=maskDS.RasterXSize    #掩膜宽度
    maskYSize = maskDS.RasterYSize  # 掩膜宽度



# raster_fn = path.join(dir_name, 'test_mask.tiff')  # 存放掩膜影像


    geoTrans = srcDS.GetGeoTransform()  # 获取地理参考6参数
    srcPro = srcDS.GetProjection()  # 获取坐标引用
    srcXSize = srcDS.RasterXSize  # 宽度
    srcYSize = srcDS.RasterYSize  # 高度
    nbands = srcDS.RasterCount  # 波段数
    # append_num = isize // 2

    # print("nbands = :", nbands)
    # data_sum = srcband.ReadAsArray(0, 0, srcXSize, srcYSize)
    data_sum_1 = srcDS.ReadAsArray()
    c1 = data_sum_1[0, :, :]
    # c1 = np.lib.pad(c1, (append_num), 'symmetric')
    c2 = data_sum_1[1, :, :]
    # c2 = np.lib.pad(c2, (append_num), 'symmetric')
    c3 = data_sum_1[2, :, :]
    # c3 = np.lib.pad(c3, (append_num), 'symmetric')
    # c4 = data_sum_1[3, :, :]
    # c4 = np.lib.pad(c4, (append_num), 'symmetric')

    # data_sum = cv2.merge([c1, c2, c3,c4])
    data_sum = cv2.merge([c1, c2, c3])
    # raster_fn = path.join(dir_name, 'test_mask.tiff')  # 存放掩膜影像
    shortname, extension = path.splitext(dir_name)
    raster_fn = shortname + "_mask_step" + str(step).zfill(2) + ".tiff"
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, srcXSize, srcYSize, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geoTrans)  # 设置掩膜的地理参考
    target_ds.SetProjection(srcPro)  # 设置掩膜坐标引用
    band = target_ds.GetRasterBand(1)  # 获取第一个波段(影像只有一个波段)

    # 用来保存去噪后的掩膜图像  __2017.0906, by xjxf _start
    target_ds_denoising = gdal.GetDriverByName('GTiff').Create(shortname + "_mask_step_" + str(step).zfill(2) + ".tiff",
                                                               srcXSize, srcYSize, 1, gdal.GDT_Byte)
    target_ds_denoising.SetGeoTransform(geoTrans)  # 设置掩膜的地理参考
    target_ds_denoising.SetProjection(srcPro)  # 设置掩膜坐标引用
    # 用来保存去噪后的掩膜图像  __2017.0906, by xjxf _end

    # band.SetNoDataValue(0)  # 将这个波段的值全设置为0
    data_target_temp = np.zeros([srcYSize, srcXSize], np.uint8)
    data_target = np.zeros([srcYSize, srcXSize], np.uint8)

    saver = tf.train.Saver()  # 默认是保存所有变量
    config = tf.ConfigProto(device_count={"CPU": 4},  # limit to num_cpu_core CPU usage
                            inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=4,
                            log_device_placement=False)
    with tf.Session(config=config) as sess:
        saver.restore(sess, model_path)

        flagY = True
        for i in range(0,srcYSize,isize ):
            if not flagY:
                break
            if i + isize > srcYSize:
                i_end = srcYSize
                flagY = False
            else:
                i_end=i+isize

            flagX = True
            for j in range(0, srcXSize, isize):
                if not flagX:
                    break
                if j + isize > srcXSize:
                    j_end=srcXSize
                    flagX = False
                else:
                    j_end=j+isize
                    # multi band to array
                    # TODO 一次性读取整张图像，每次截取3×3区域进行计算并将结果暂存至数组，最后统一写入磁盘
                    # for band in range(nbands):
                    #     band += 1
                    # print(band)
                    # print "[ GETTING BAND ]: ", band
                    # srcband = srcDS.GetRasterBand(band)  # 获取该波段
                    # if srcband is None:
                    #     continue

                    # Read raster as arrays 类似RasterIO（C++）
                    # dataraster = srcband.ReadAsArray(j, i, isize, isize, isize, isize)

                dataraster = (data_sum[i:i_end, j:j_end])
                dataraster=dataraster*(1./255)-0.5
                dataraster_new=np.lib.pad(dataraster,((0,isize-(i_end-i)),(0,isize-(j_end-j)),(0,0)),'constant',constant_values=0)
                dataraster_new = np.array(dataraster_new)

                data = dataraster_new.reshape([-1, img_size])
                ori_y = i
                ori_x = j


                pred1 = sess.run(pred_new, feed_dict={x: data, keep_prob: 1.})
                value_1 = np.argmax(pred1, axis=1)

                #重新生成掩膜图像，2017.10.11,by xjxf __start
                mask_value=value_1.reshape([isize,isize])
                # 重新生成掩膜图像，2017.10.11,by xjxf __end

                # 将块写入掩膜图像，2017.10.11,by xjxf __start
                data_target_temp[ori_y:i_end,ori_x:j_end]=mask_value[0:i_end-ori_y,0:j_end-ori_x]
                # 将块写入掩膜图像，2017.10.11,by xjxf __end
                # print("ori_x:",str(ori_x),"ori_y",str(ori_y))
                # target_ds.FlushCache()  # 将数据写入文件

        # data_target=data_target*255
        # #将结果重新映射 2017.10.18,by xjxf  _start
        # result_row,result_col=data_target_temp.shape
        # for ii in range(result_row):
        #     for jj in range(result_col):
        #         data_target[ii][jj]=bid_dict[data_target_temp[ii][jj]]
        # # 将结果重新映射 2017.10.18,by xjxf  _end

        data_target=data_target_temp
        target_ds.GetRasterBand(1).WriteArray(data_target, 0, 0)
        target_ds.FlushCache()
        # 对生成掩膜滤波（中值滤波），并保存__2017.0906,by xjxf__start
        ori_img = target_ds.GetRasterBand(1)
        ori_img_data = ori_img.ReadAsArray()
        blur = cv2.medianBlur(ori_img_data, 5)
        target_ds_denoising.GetRasterBand(1).WriteArray(blur, 0, 0)
        target_ds_denoising.FlushCache()
        # 对生成掩膜滤波（中值滤波），并保存__2017.0906,by xjxf__end


        # 计算结果精度__strat
        band_new_1 = target_ds.GetRasterBand(1)
        band_new_2 = maskDS.GetRasterBand(1)
        accuracy_sum = 0
        img_1 = band_new_1.ReadAsArray()


        img_2 = band_new_2.ReadAsArray()
        img_2[img_1 is None] = 0

        sum_num=0
        for i_new_1 in range(0, min(srcXSize,maskXSize)):
            for j_new_1 in range(0, min(srcYSize,maskYSize)):
                if img_2[j_new_1,i_new_1]>=0:
                    sum_num+=1
                    if img_1[j_new_1, i_new_1] == img_2[j_new_1, i_new_1]:
                        # if band_new_1.ReadAsArray(i_new_1, j_new_1).all() == band_new_2.ReadAsArray(i_new_1, j_new_1).all():
                        accuracy_sum = accuracy_sum + 1
                    else:
                        accuracy_sum = accuracy_sum - 1
        # accuracy_mean = accuracy_sum / (srcYSize * srcXSize)
        accuracy_mean = accuracy_sum / sum_num
        print("the accuracy is %10.8f:" % accuracy_mean)
        # 计算结果精度__end


        # target_ds.FlushCache()  # 将数据写入文件
        endtime = datetime.datetime.now()
        print("endtime: ", endtime)
        print("time used in seconds: ", (endtime - starttime).seconds)

    fpa.close()

if __name__ == '__main__':
    polypro(sys.argv)
