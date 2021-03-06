# -*- coding: UTF-8 -*-

'''
--------------集群版----------------
集成
3波段城市道路模型 400x400x3  400x400
4波段水体模型 9x9x4 1x1

还没联调（没有调试）

train=1 训练
train=-1 推理
nohup python mnist_dist.py --ps_hosts=10.0.100.14:2220 --worker_hosts=10.0.100.14:2221 --job_name="ps" --task_index=0 --start_index=0 --end_index=3717 &
nohup python mnist_dist.py --ps_hosts=10.0.100.14:2220 --worker_hosts=10.0.100.14:2221 --job_name="worker" --task_index=0 --start_index=0 --end_index=3717 &
python mnist_dist.py \
--ps_hosts=10.0.100.14:2220 \
--worker_hosts=10.0.100.15:2221 \
--job_name="ps" \
--train=1 \
--task_index=0 \
--start_index=0 \
--end_index=1 \
--image_size=400 \
--image_channel=3 \
--epochs=2 \
--batch_size=1 \
--display_step=1 \
--n_class=3 \
--dropout=0.8 \
--image_path='" \
--mask_path="" \
--logdir=" " \
--model_name="save_net.ckpt" \
--in_image_dir="" \
--in_image_path="" \
--save_image_path="" 

# train
python mnist_dist.py \
--ps_hosts=10.0.100.14:2220 \
--worker_hosts=10.0.100.15:2221 \
--job_name="ps" \
--train=1 \
--task_index=0 \
--image_size=400 \
--image_channel=3 \
--epochs=2 \
--batch_size=1 \
--display_step=1 \
--n_class=3 \
--dropout=0.8 \
--image_path='" \
--mask_path="" \
--logdir=" " \
--model_name="save_net.ckpt" \

# inference
python mnist_dist.py \
--ps_hosts=10.0.100.14:2220 \
--worker_hosts=10.0.100.15:2221 \
--job_name="worker" \
--train=-1 \
--task_index=0 \
--start_index=0 \
--end_index=2 \
--image_size=400 \
--image_channel=3 \
--n_class=3 \
--logdir=" " \
--model_name="save_net.ckpt" \
--in_image_dir="" \
--in_image_path="" \
--save_image_path=""
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

try:
    from osgeo import gdal, ogr
except:
    import gdal, ogr

import os
from os import path
import datetime
# import Three_model

from osgeo.gdalconst import *
import glob

start_time = datetime.datetime.now()
print("startTime: ", start_time)


def create_pickle_train(image_path, mask_path, img_pixel=9, channels=4):
    m = 0

    image_data = Multiband2Array(image_path, channels)
    print("data_matrix_max= ", image_data.max())
    print("data_matrix_min= ", image_data.min())
    # mask_data = cv2.split(cv2.imread(mask_path))[0] / 255
    mask_data = Multiband2Array(mask_path, channels) / 255

    x_size, y_size = image_data.shape[:2]

    data_list = []

    for i in range(0, x_size - img_pixel + 1, img_pixel // 2):  # 文件夹下的文件名
        if i + img_pixel > x_size:
            i = x_size - img_pixel - 1
        for j in range(0, y_size - img_pixel + 1, img_pixel // 2):
            if j + img_pixel > y_size:
                j = y_size - img_pixel - 1
            cropped_data = image_data[i:i + img_pixel, j:j + img_pixel]
            data1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))  # 展成一行
            train_label = mask_data[i + img_pixel // 2, j + img_pixel // 2]
            data2 = np.append(data1, train_label)[np.newaxis, :]  # 数据+标签

            data_list.append(data2)

            m += 1

            if m % 10000 == 0:
                print(datetime.datetime.now(), "compressed {number} images".format(number=m))
                data_matrix = np.array(data_list, dtype=int)

                data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * channels + 1)))
                return data_matrix

    print(len(data_list))
    print(m)

    data_matrix = np.array(data_list, dtype=int)

    data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * channels + 1)))

    """
    with gzip.open('D:/train_data_64.pkl', 'ab') as writer:  # 以压缩包方式创建文件，进一步压缩文件
        pickle.dump(data_matrix, writer)  # 数据存储成pickle文件
    """
    return data_matrix  # shape [none,9*9*4+1]


def create_pickle_train2(image_path, mask_path, img_pixel=400, channels=4):
    m = 0
    compress_count = 0  # 增加一个值，来返回压缩了几次  by bxjxf
    # num_img+=1
    # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # mask_img= gdal.Open(mask_path, gdal.GA_ReadOnly)  # 只读方式打开原始影像
    # srcXSize = mask_img.RasterXSize  # 宽度
    # srcYSize = mask_img.RasterYSize  # 高度
    # band=mask_img.GetRasterBand(1)
    # mask_data=band.ReadAsArray(0,0,srcYSize,srcXSize)
    step = img_pixel // 3
    '''
    mask_img=gdal.Open(mask_path)
    mask_data_temp=mask_img.ReadAsArray()
    # print(mask_data.shape)
    row_num,col_num=mask_data_temp.shape
    mask_data=np.zeros([row_num,col_num])
    mask_data=mask_data_temp
    '''
    mask_data = Multiband2Array(mask_path, 1)
    if np.max(mask_data) == 255: mask_data = mask_data / 255
    # mask_data=mask_data_temp
    # #将图像中的编码映射成序列数   2017.10.16,by xjxf  __start
    # for i_1 in range(row_num):
    #     for j_1 in range(col_num):
    #         if mask_data_temp[i_1,j_1] ==2 or mask_data_temp[i_1,j_1]==50:
    #             mask_data[i_1,j_1]=1
    #         elif mask_data_temp[i_1,j_1]==3 or mask_data_temp[i_1,j_1]==54:
    #             mask_data[i_1,j_1]=2
    #         else:
    #             mask_data[i_1, j_1] = 0
    # mask_data_new=mask_data.reshape([row_num,col_num])
    # cv2.imwrite("xjxf.tif",mask_data_new)
    # print(num_img)
    # # 将图像中的编码映射成序列数   2017.10.16,by xjxf  __end

    image_data = Multiband2Array(image_path, channels)

    # mask_data = cv2.split(cv2.imread(mask_path))[0]

    x_size, y_size = image_data.shape[:2]

    data_list = []
    flag_x = True
    flag_y = True
    # print(len(data_list))

    for i in range(0, x_size - img_pixel + 1, step):  # 文件夹下的文件名
        i_end = i + img_pixel
        if i + img_pixel > x_size:
            # i = x_size - img_pixel - 1
            i_end = x_size
            flag_x = False

        flag_y = False
        for j in range(0, y_size - img_pixel + 1, step):
            j_end = j + img_pixel
            if j + img_pixel > y_size:
                #     j = y_size - img_pixel - 1
                j_end = y_size
                flag_y = False

            cropped_data_temp = image_data[i:i_end, j:j_end]
            # 对截取的样本做扩充, 2017.10.24, by xjxf __start

            cropped_data = np.lib.pad(cropped_data_temp,
                                      ((0, img_pixel - (i_end - i)), (0, img_pixel - (j_end - j)), (0, 0)), 'constant',
                                      constant_values=0)
            # 对截取的样本做扩充, 2017.10.24, by xjxf __end


            data_1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))  # 展成一行
            # cropped_data_2 = image_data[i:i + img_pixel, j:j + img_pixel, 1]
            # data_2 = cropped_data_2.reshape((-1, img_pixel * img_pixel ))  # 展成一行
            # cropped_data_3 = image_data[i:i + img_pixel, j:j + img_pixel, 2]
            # data_3 = cropped_data_3.reshape((-1, img_pixel * img_pixel ))  # 展成一行
            cropped_mask_data_temp = mask_data[i:i_end, j:j_end]
            # 对截取的样本做扩充, 2017.10.24, by xjxf __start
            cropped_mask_data = np.lib.pad(cropped_mask_data_temp,
                                           ((0, img_pixel - (i_end - i)), (0, img_pixel - (j_end - j))), 'constant',
                                           constant_values=0)
            # 对截取的样本做扩充, 2017.10.24, by xjxf __end
            train_label = cropped_mask_data.reshape((-1, img_pixel * img_pixel))

            # data2 = np.append(data_1[np.newaxis,:], data_2[np.newaxis,:])
            # data2=np.append(data2,data_3[np.newaxis,:])

            data2 = np.append(data_1, train_label)[np.newaxis, :]

            # if train_label==0 or train_label==1 or train_label==3 or train_label==5:    #去除标签是其他的样本
            # if train_label==1:    #去除标签是其他的样本
            #     print("hello")
            data_list.append(data2)
            m += 1

            if m % 10000 == 0:
                print(datetime.datetime.now(), "compressed {number} images".format(number=m))
                data_matrix = np.array(data_list, dtype=int)

                data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * (channels + 1))))
                return data_matrix

    print(len(data_list))
    print(m)

    data_matrix = np.array(data_list, dtype=int)

    data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * (channels + 1))))

    return data_matrix
    '''
            # if m % 10000 == 0: print(datetime.datetime.now(), "compressed {number} images".format(number=m))
            # 每到一百万张采样图片时，保存压缩一次到硬盘  ___start_by xjxf
            if m%1000000==0:
                compress_count += 1
                print("第"+str(compress_count)+"次压缩")
                data_matrix = np.array(data_list, dtype=int)
                data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * (channels + 1))))
                with gzip.open(filename, 'ab') as writer:  # 以压缩包方式创建文件，进一步压缩文件
                    pickle.dump(data_matrix, writer)  # 数据存储成pickle文件
                data_list=[]
            # 每到一万张采样图片时，保存压缩一次到硬盘  ___end_by xjxf
    print(m)
    #将最后一部分取样也做压缩，__start,by xjxf
    if len(data_list)> 0:
        data_matrix = np.array(data_list, dtype=int)
        data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * (channels + 1))))
        with gzip.open(filename,'ab') as writer:  # 以压缩包方式创建文件，进一步压缩文件
            pickle.dump(data_matrix, writer)  # 数据存储成pickle文件
        compress_count+=1
    return compress_count
    # 将最后一部分取样也做压缩，__end,by xjxf
    '''


def Multiband2Array(path, channels):
    src_ds = gdal.Open(path)
    if src_ds is None:
        print('Unable to open %s' % path)
        sys.exit(1)

    xcount = src_ds.RasterXSize  # 宽度
    ycount = src_ds.RasterYSize  # 高度
    ibands = src_ds.RasterCount  # 波段数

    # print "[ RASTER BAND COUNT ]: ", ibands
    # if ibands==4:ibands=3
    ibands = min(channels, ibands)
    for band in range(ibands):
        band += 1
        # print "[ GETTING BAND ]: ", band
        srcband = src_ds.GetRasterBand(band)  # 获取该波段
        if srcband is None:
            continue

        # Read raster as arrays 类似RasterIO（C++）
        dataraster = srcband.ReadAsArray(0, 0, xcount, ycount).astype(np.float16)
        if ibands == 1: return dataraster.reshape((ycount, xcount))
        if band == 1:
            data = dataraster.reshape((ycount, xcount, 1))
        else:
            # 将每个波段的数组很并到一个3维数组中
            data = np.append(data, dataraster.reshape((ycount, xcount, 1)), axis=2)

    return data


def next_batch(data, batch_size, flag, img_pixel=3, channels=4):
    global start_index  # 必须定义成全局变量
    global second_index  # 必须定义成全局变量

    if 1 == flag:
        start_index = 0
    # start_index = 0
    second_index = start_index + batch_size

    if second_index > len(data):
        second_index = len(data)

    data1 = data[start_index:second_index]
    # print('start_index', start_index, 'second_index', second_index)

    start_index = second_index
    if start_index >= len(data):
        start_index = 0

    # 将每次得到batch_size个数据按行打乱
    index = [i for i in range(len(data1))]  # len(data1)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    data1 = data1[index]

    # 提取出数据和标签
    img = data1[:, 0:img_pixel * img_pixel * channels]

    # img = img * (1. / img.max) - 0.5
    img = img * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
    img = img.astype(np.float32)  # 类型转换

    label = data1[:, img_pixel * img_pixel * channels:]
    label = label.reshape([-1, 1])
    label = label.astype(int)  # 类型转换

    return img, label


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    # 从标量类标签转换为一个one-hot向量
    num_labels = labels_dense.shape[0]  # label的行数
    index_offset = np.arange(num_labels) * num_classes
    # print index_offset
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def dense_to_one_hot2(labels_dense, num_classes):
    labels_dense = np.array(labels_dense, dtype=np.uint8).reshape([-1, 1])  # [160000,1]
    num_labels = labels_dense.shape[0]  # 标签个数 160000
    labels_one_hot = np.zeros((num_labels, num_classes), np.uint8)  # [160000,3]
    for i, itenm in enumerate(labels_dense):
        labels_one_hot[i, itenm] = 1
        # 如果labels_dense不是int类型，itenm就不是int，此时做数组的切片索引就会报错，
        # 数组索引值必须是int类型，也可以 int(itenm) 强制转成int
        # labels_one_hot[i, :][itenm] = 1
    return labels_one_hot


# 参数设置

tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

tf.app.flags.DEFINE_integer("train", 1, "1 train,-1 inference")

tf.app.flags.DEFINE_integer("model", 1, "1 3-band urban road model;2 4-band water body model")
# tf.app.flags.DEFINE_integer("image_size", 400, "size of image")
# tf.app.flags.DEFINE_integer("image_channel", 3, "channel of image")
# tf.app.flags.DEFINE_integer("n_class", 3, "The sample classification number")

tf.app.flags.DEFINE_integer("epochs", 10, "Training the number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 128, "Each training sample size")
tf.app.flags.DEFINE_integer("display_step", 1, "How many times each iteration to print the results")
tf.app.flags.DEFINE_integer("dropout", 0.8, "Droupout retention")


tf.app.flags.DEFINE_string("image_path", "gtdata:///users/xiaoshaolin/tensorflow/train_data/train_img_01.tif", "image path")
tf.app.flags.DEFINE_string("mask_path", "gtdata:///users/xiaoshaolin/tensorflow/train_data/train_img_mask_01.tif", "mask path")

tf.app.flags.DEFINE_string("logdir", "hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/model", "Model store address")
tf.app.flags.DEFINE_string("model_name", "save_net.ckpt", "Model name")


tf.app.flags.DEFINE_string("in_image_dir", "hdfs://dm01-08-01.tjidc.dcos.com:8020/nanlin/china_q3/", "The image directory for inference")
tf.app.flags.DEFINE_string("in_image_path", "./tif.log", "The image directory for inference")
tf.app.flags.DEFINE_integer("start_index", 0, "Index of start within img for inference")
tf.app.flags.DEFINE_integer("end_index", 3717, "Index of end within img for inference")

tf.app.flags.DEFINE_string("save_image_path", "./image_walter/", "The image save path for inference")

FLAGS = tf.app.flags.FLAGS

# 超参数

# Parameters
training_epochs = FLAGS.epochs
if FLAGS.model==1:
    batch_size=1
    isize = 400
    img_channel = 3
    img_pixel = 400
    img_pixel_h = 400
    img_pixel_w = 400
    label_cols = 3
else:
    batch_size = FLAGS.batch_size
    isize = 9
    img_channel = 4
    img_pixel = 9
    img_pixel_h = 9
    img_pixel_w = 9
    label_cols = 2

display_step = FLAGS.display_step
channels = img_channel

# Network Parameters
img_size = isize * isize * channels

dropout = FLAGS.dropout
train = FLAGS.train  # 1 tarin ;-1 inference

class Input(object):
    def __init__(self):
        # self.image_path = 'gtdata:///users/xiaoshaolin/tensorflow/train_data/train_img_01.tif'
        # self.mask_path = 'gtdata:///users/xiaoshaolin/tensorflow/train_data/train_img_mask_01.tif'

        self.image_path = FLAGS.image_path
        self.mask_path = FLAGS.mask_path

    def split_train_and_test(self):
        self.data = create_pickle_train2(self.image_path, self.mask_path, isize, channels)
        np.random.shuffle(self.data)

        # 选取0.3测试数据与0.7训练数据
        # self.train_data = self.data[:int(len(self.data) * 0.7)]
        # self.test_data = self.data[int(len(self.data) * 0.7):]

        # return [self.data,self.train_data,self.test_data,len(self.data)]
        return [self.data, len(self.data)]

    def split_train_and_test2(self):
        self.data = create_pickle_train(self.image_path, self.mask_path, isize, channels)
        np.random.shuffle(self.data)

        # 选取0.3测试数据与0.7训练数据
        # self.train_data = self.data[:int(len(self.data) * 0.7)]
        # self.test_data = self.data[int(len(self.data) * 0.7):]

        # return [self.data,self.train_data,self.test_data,len(self.data)]
        return [self.data, len(self.data)]


if train == 1:
    if FLAGS.model==1:
        data, img_nums = Input().split_train_and_test()
    if FLAGS.model==2:
        data, img_nums = Input().split_train_and_test2()


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # cluster = tf.train.ClusterSpec({"local": ps_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    print("Cluster job: %s, task_index: %d, target: %s" % (FLAGS.job_name, FLAGS.task_index, server.target))
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            # -------------------------------------------------------------#

            x = tf.placeholder(tf.float32, [None, img_size])  # mnist data image of shape 28*28=784
            y = tf.placeholder(tf.float32, [None, label_cols])  #
            keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

            global_step = tf.Variable(0)

            if train == 1:
                # ---------设置动态学习效率
                # Constants describing the training process.
                # MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
                NUM_EPOCHS_PER_DECAY = batch_size  # Epochs after which learning rate decays.
                LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.
                INITIAL_LEARNING_RATE = 0.00015  # Initial learning rate.

                # global_step = training_epochs * (img_nums // batch_size)  # Integer Variable counting the number of training steps     # //是整数除法
                # Variables that affect learning rate.
                num_batches_per_epoch = int(img_nums / batch_size)
                # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                decay_steps = int(num_batches_per_epoch * 2)
                # decay_steps = int(2)
                # Decay the learning rate exponentially based on the number of steps.
                learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                                           global_step,
                                                           decay_steps,
                                                           LEARNING_RATE_DECAY_FACTOR,
                                                           staircase=True)
            if train == -1:
                learning_rate = 1e-5

            def model_1():
                layer_dict = {
                    'wc1_1': [3, 3, channels, 64],
                    'wc1_2': [3, 3, 64, 64],

                    'wc2_1': [3, 3, 64, 128],
                    'wc2_2': [3, 3, 128, 128],

                    'wc3_1': [3, 3, 128, 256],
                    'wc3_2': [3, 3, 256, 256],
                    'wc3_3': [3, 3, 256, 256],

                    'wc4_1': [3, 3, 256, 512],
                    'wc4_2': [3, 3, 512, 512],
                    'wc4_3': [3, 3, 512, 512],

                    'wc5_1': [3, 3, 512, 512],
                    'wc5_2': [3, 3, 512, 512],
                    'wc5_3': [3, 3, 512, 512],

                    'wf_6': [7, 7, 512, 4096],
                    'wf_7': [1, 1, 4096, 4096],

                    'w_out': [1, 1, 4096, label_cols],

                    'up_sample_p_4': [1, 1, 512, label_cols],

                    'up_sample_p_3': [1, 1, 256, label_cols],

                    'up_sample_p_2': [1, 1, 128, label_cols]

                }

                # 定义一个简化卷积层操作的函数  2017.09.29 by xhxf__start
                def conv_relu(bottom, name, stride=1):
                    # bottom_new=np.lib.pad(bottom,(pad),"contant",constant_values=0)
                    layer_name = name
                    layer_pram = layer_dict[name]
                    with tf.variable_scope(layer_name):
                        with tf.variable_scope('weights'):
                            weights_1 = tf.get_variable(name + '_kernel', shape=layer_pram,
                                                        initializer=tf.random_uniform_initializer)  # glorot_uniform_initializer
                            tf.summary.histogram(layer_name + '/weights', weights_1)  # 可视化观察变量  by xjxf, 2017.10.11
                        with tf.variable_scope('biases'):
                            biases_1 = tf.get_variable(name + 'biases', shape=layer_pram[3],
                                                       initializer=tf.random_uniform_initializer)
                            tf.summary.histogram(layer_name + '/biasses', biases_1)  # 可视化观察变量  by xjxf, 2017.10.11

                    conv = tf.nn.conv2d(bottom, weights_1, strides=[1, stride, stride, 1], padding="SAME")
                    conv = tf.nn.bias_add(conv, biases_1)
                    return conv, tf.nn.relu(conv)

                # 定义一个简化卷积层操作的函数  2017.09.29 by xhxf__ned

                # 定义一个简化池化层操作过程的函数  2017.09.30 by xjjxf  _start
                def max_pool(bottom, k=2):
                    return tf.nn.max_pool(bottom, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")
                    # return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

                # 定义一个简化池化层操作过程的函数  2017.09.30 by xjjxf  _start

                # Create model
                def conv_net(x, dropout):
                    # Reshape input picture  x shape [batch,400*400*3]
                    x = tf.reshape(x, shape=[-1, img_pixel_h, img_pixel_w, channels])  # [batch,400,400,3]

                    # Convolution Layer
                    conv1_1, relu1_1 = conv_relu(x, 'wc1_1')  # [batch,400,400,64]
                    conv1_2, relu1_2 = conv_relu(relu1_1, 'wc1_2')  # [batch,400,400,64]
                    pool_1 = max_pool(relu1_2)  # [batch,200,200,64]

                    conv2_1, relu2_1 = conv_relu(pool_1, 'wc2_1')  # [batch,200,200,128]
                    conv2_2, relu2_2 = conv_relu(relu2_1, 'wc2_2')  # [batch,200,200,128]
                    pool_2 = max_pool(relu2_2)  # [batch,100,100,128]

                    conv3_1, relu3_1 = conv_relu(pool_2, 'wc3_1')  # [batch,100,100,256]
                    conv3_2, relu3_2 = conv_relu(relu3_1, 'wc3_2')  # [batch,100,100,256]
                    conv3_3, relu3_3 = conv_relu(relu3_2, 'wc3_3')  # [batch,100,100,256]
                    pool_3 = max_pool(relu3_3)  # [batch,50,50,256]

                    conv4_1, relu4_1 = conv_relu(pool_3, 'wc4_1')  # # [batch,50,50,512]
                    conv4_2, relu4_2 = conv_relu(relu4_1, 'wc4_2')  # [batch,50,50,512]
                    conv4_3, relu4_3 = conv_relu(relu4_2, 'wc4_3')  # [batch,50,50,512]
                    pool_4 = max_pool(relu4_3)  # [batch,25,25,512]

                    conv5_1, relu5_1 = conv_relu(pool_4, 'wc5_1')  # [batch,25,25,512]
                    conv5_2, relu5_2 = conv_relu(relu5_1, 'wc5_2')  # [batch,25,25,512]
                    conv5_3, relu5_3 = conv_relu(relu5_2, 'wc5_3')  # [batch,25,25,512]
                    pool_5 = max_pool(relu5_3)  # # [batch,13,13,512]

                    fc_6, relu_fc6 = conv_relu(pool_5, 'wf_6')  # [batch,13,13,4096]
                    relu_fc6 = tf.nn.dropout(relu_fc6, dropout)  # [batch,13,13,4096]

                    fc_7, relu_fc7 = conv_relu(relu_fc6, 'wf_7')  # [batch,13,13,4096]
                    relu_fc7 = tf.nn.dropout(relu_fc7, dropout)  # [batch,13,13,4096]

                    out_1, relu_out1 = conv_relu(relu_fc7, 'w_out')  # [batch,13,13,3]

                    # fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])  # [None,15*15*64]
                    # fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
                    # fc1 = tf.nn.relu(fc1)
                    # # Apply Dropout
                    # fc1 = tf.nn.dropout(fc1, dropout)
                    #
                    # # Output, class prediction
                    # out_1 = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
                    # 上采样至pool_4的大小  2017.10.13,by xjxf   __start
                    out_end_1 = tf.nn.conv2d_transpose(out_1,
                                                       tf.get_variable("up_sample_1",
                                                                       shape=[4, 4, label_cols, label_cols],
                                                                       initializer=tf.random_uniform_initializer()),
                                                       output_shape=[batch_size, img_pixel_h // 16,
                                                                     img_pixel_w // 16,
                                                                     label_cols],
                                                       strides=[1, 2, 2, 1],
                                                       padding="SAME")  # [batch_size,img_pixel_h//16,img_pixel_w//16,label_cols] 即[batch,25,25,3]
                    # 上采样至pool_4的大小  2017.10.13,by xjxf   __end

                    # 让pool_4卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start
                    upsample_pool4, upsample_pool4_relu = conv_relu(pool_4, 'up_sample_p_4')  # [batch,25,25,3]
                    out_upsample_pool4 = upsample_pool4 + out_end_1  # [batch,25,25,3]
                    out_end_2 = tf.nn.conv2d_transpose(out_upsample_pool4,
                                                       tf.get_variable("up_sample_4",
                                                                       shape=[4, 4, label_cols, label_cols],
                                                                       initializer=tf.random_uniform_initializer()),
                                                       output_shape=[batch_size, img_pixel_h // 8, img_pixel_w // 8,
                                                                     label_cols],
                                                       strides=[1, 2, 2, 1],
                                                       padding='SAME')  # [batch_size,img_pixel_h//8,img_pixel_w//8,label_cols] 即 [batch,50,50,3]
                    # 让pool_4卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start

                    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start
                    upsample_pool3, upsample_pool3_relu = conv_relu(pool_3, 'up_sample_p_3')  # [batch,50,50,3]
                    out_upsample_pool3 = upsample_pool3 + out_end_2  # [batch,50,50,3]
                    out_end_3 = tf.nn.conv2d_transpose(out_upsample_pool3,
                                                       tf.get_variable("up_sample_3",
                                                                       shape=[4, 4, label_cols, label_cols],
                                                                       initializer=tf.random_uniform_initializer()),
                                                       output_shape=[batch_size, img_pixel_h // 4, img_pixel_w // 4,
                                                                     label_cols],
                                                       strides=[1, 2, 2, 1],
                                                       padding='SAME')  # [batch_size, img_pixel_h//4, img_pixel_w//4, label_cols] 即 [batch,100,100,3]
                    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start

                    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.16,by xjxf  __start
                    upsample_pool2, upsample_pool2_relu = conv_relu(pool_2, 'up_sample_p_2')  # [batch,100,100,3]
                    out_upsample_pool2 = upsample_pool2 + out_end_3  # [batch,100,100,3]
                    out_end_4 = tf.nn.conv2d_transpose(out_upsample_pool2,
                                                       tf.get_variable("up_sample_2",
                                                                       shape=[8, 8, label_cols, label_cols],
                                                                       initializer=tf.random_uniform_initializer()),
                                                       output_shape=[batch_size, img_pixel_h, img_pixel_w,
                                                                     label_cols],
                                                       strides=[1, 4, 4, 1],
                                                       padding='SAME')  # [batch_size, img_pixel_h, img_pixel_w, label_cols] 即 [batch,400,400,3]
                    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.16,by xjxf  __start
                    return out_end_4  # [batch,400,400,3]

                # Construct model
                pred = conv_net(x, keep_prob)
                pred_new = tf.reshape(pred, [-1, label_cols])

                return pred_new

            def model_2():
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
                    x = tf.reshape(x, shape=[-1, img_pixel, img_pixel, channels])  # 9x9x4

                    # Convolution Layer
                    conv1 = conv2d(x, weights['wc1'], biases['bc1'])  # 图像9x9x64
                    # Max Pooling (down-sampling)
                    conv1 = maxpool2d(conv1, k=2)  # 图像 5x5x64

                    # Convolution Layer
                    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])  # 图像 5x5x128
                    # Max Pooling (down-sampling)
                    conv2 = maxpool2d(conv2, k=2)  # 图像 3x3x128

                    # Convolution Layer
                    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])  # 图像 3x3x256
                    # Max Pooling (down-sampling)
                    # conv3 = maxpool2d(conv3, k=2)  # 图像 2x2x256

                    # Fully connected layer
                    # Reshape conv2 output to fit fully connected layer input
                    fc1 = tf.reshape(conv2,
                                     [-1, weights['wd1'].get_shape().as_list()[0]])  # [None,2*2*256]-->[None,1024]
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
                    'wc1': tf.Variable(tf.random_normal([5, 5, channels, 64])),
                # 5X5的卷积模板   #tf.random_normal是从正态分布中输出随机数
                    # 5x5 conv, 32 inputs, 64 outputs
                    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
                    # 5x5 conv, 128 inputs, 128 outputs
                    # 'wc3': tf.Variable(tf.random_normal([5, 5, 128, 256])),
                    # fully connected, 7*7*64 inputs, 1024 outputs
                    'wd1': tf.Variable(tf.random_normal([(1 + img_pixel // 4) * (1 + img_pixel // 4) * 128, 1024])),
                    # 1024 inputs, 10 outputs (class prediction)
                    'out': tf.Variable(tf.random_normal([1024, label_cols]))
                }

                biases = {
                    'bc1': tf.Variable(tf.random_normal([64])),
                    'bc2': tf.Variable(tf.random_normal([128])),
                    # 'bc3': tf.Variable(tf.random_normal([256])),
                    'bd1': tf.Variable(tf.random_normal([1024])),
                    'out': tf.Variable(tf.random_normal([label_cols]))
                }

                # Construct model
                pred = conv_net(x, weights, biases, keep_prob)

                return pred


            if FLAGS.model == 1:
                pred_new=model_1()
            if FLAGS.model == 2:
                pred_new=model_2()
            # 可视化cost值,by xjxf 2017.10.11 __start
            with tf.name_scope('cost'):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred_new))
                # tf.summary.scalar('cost', cost)
            # 可视化cost值,by xjxf 2017.10.11 __end
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,
                                                                                     global_step=global_step)
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
            # Evaluate model
            correct_pred = tf.equal(tf.argmax(pred_new, 1), tf.argmax(y, 1))
            # 可视化cost值,by xjxf 2017.10.11 __start
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                # tf.summary.scalar('accuracy', accuracy)
            # 可视化cost值,by xjxf 2017.10.11 __end

            # summary_op = tf.summary.merge_all()

            # 初始化所有的op
            init = tf.global_variables_initializer()

            saver = tf.train.Saver()

        # logdir = "hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/model"
        logdir=FLAGS.logdir
        # Create a "Supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=logdir,
                                 init_op=init,
                                 # summary_op=None,
                                 saver=saver,
                                 # saver=None, # None 不自动保存模型
                                 # recovery_wait_secs=1,
                                 global_step=global_step,
                                 stop_grace_secs=300,
                                 save_model_secs=10,
                                 checkpoint_basename=FLAGS.model_name)

        # The supervisor takes care of session initialization and restoring from
        # a checkpoint.
        sess = sv.prepare_or_wait_for_session(server.target)

        # Start queue runners for the input pipelines (if ang).
        sv.start_queue_runners(sess)
        if train == 1:
            # cost, pred, optimizer, accuracy = m
            def Train():
                total_batch = int(img_nums / batch_size)

                for epoch in range(training_epochs):
                    np.random.shuffle(data)
                    avg_cost = 0.
                    flag = 1
                    for i in range(total_batch):
                        img, label = next_batch(data, batch_size, flag, img_pixel=isize, channels=img_channel)
                        flag = 0
                        batch_xs = img.reshape([-1, img_size])

                        batch_ys = dense_to_one_hot2(label[:, np.newaxis],
                                                     label_cols)  # 生成多列标签   问题6，生成多列标签是干什么呢？   by xjxf
                        # Run optimization op (backprop) and cost op (to get loss value)
                        print('batch_ys', batch_ys.shape)
                        exit(-1)
                        _, c, p, gl_step = sess.run([optimizer, cost, pred_new, global_step],
                                                    feed_dict={x: batch_xs,
                                                               y: batch_ys,
                                                               keep_prob: dropout})

                        # if i % 20 == 0: print('global_step', gl_step, 'cost', c)
                        # Compute average loss
                        avg_cost += c / total_batch

                    # Display logs per epoch step
                    if (epoch + 1) % display_step == 0:
                        print("time: ", datetime.datetime.now())
                        print("Epoch:", '%04d' % (epoch + 1),
                              "cost:", "{:.9f}".format(avg_cost),
                              'accuracy:',
                              sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0}))

                print("Optimization Finished!")
                # end_time = datetime.datetime.now()
                # print("end_time: ", end_time)
                # print("time used: ", end_time - start_time)

            while not sv.should_stop():
                Train()
                break
            end_time = datetime.datetime.now()
            print("总时间: ", end_time - start_time)
            sess.close()
            exit()

        if train == -1:  # inference
            # _, pred, _, _ = m
            def Inference(task_index):
                # img_paths = ['gtdata:///users/xiaoshaolin/tensorflow/train_data/train_img_01.tif',
                #              'gtdata:///users/xiaoshaolin/tensorflow/train_data/train_img_02.tif']
                # img_paths=['HDFS://dm01-08-01.tjidc.dcos.com:8020/nanlin/china_q3/L15-1792E-1340N.tif',
                #            'HDFS://dm01-08-01.tjidc.dcos.com:8020/nanlin/china_q3/L15-1792E-1339N.tif']

                # img_paths = ['HDFS://dm01-08-01.tjidc.dcos.com:8020/user/root/images/19.tif',
                #              'HDFS://dm01-08-01.tjidc.dcos.com:8020/user/root/images/20.tif']
                img_paths = []

                [img_paths.append(FLAGS.in_image_dir + i.replace('\n', ''))
                 for i in open(FLAGS.in_image_path)]  # 从hdfs上读取文件tile.log

                # m = 1
                # isizes = m * isize

                # 为了支持中文路径，请添加下面这句代码
                gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")

                gdal.AllRegister()  # 注册驱动
                ogr.RegisterAll()

                end = min(FLAGS.end_index, len(img_paths))
                for ix, dir_name in enumerate(img_paths[FLAGS.start_index:end]):
                    print('时间：', datetime.datetime.now(), "正在处理第", ix, '张')

                    srcDS = gdal.Open(dir_name, GA_ReadOnly)  # 只读方式打开原始影像

                    geoTrans = srcDS.GetGeoTransform()  # 获取地理参考6参数
                    srcPro = srcDS.GetProjection()  # 获取坐标引用
                    srcXSize = srcDS.RasterXSize  # 宽度
                    srcYSize = srcDS.RasterYSize  # 高度
                    nbands = srcDS.RasterCount  # 波段数

                    # dBuf_1 = np.zeros([srcYSize, srcXSize], np.uint8)  # 整个掩膜的缓存
                    # band.SetNoDataValue(0)  # 将这个波段的值全设置为0
                    data_target_temp = np.zeros([srcYSize, srcXSize], np.uint8)
                    # data_target = np.zeros([srcYSize, srcXSize], np.uint8)

                    start = datetime.datetime.now()
                    print('session start:', datetime.datetime.now())

                    # 一次性读取图像所有数据
                    nbands = min(channels, nbands)
                    for band in range(nbands):
                        # band=int(band)
                        band += 1
                        # print "[ GETTING BAND ]: ", band
                        srcband = srcDS.GetRasterBand(band)  # 获取该波段
                        if srcband is None:
                            continue

                        # Read raster as arrays 类似RasterIO（C++）
                        dataraster = srcband.ReadAsArray(0, 0, srcXSize, srcYSize, srcXSize,
                                                         srcYSize)  # .astype(np.uint8)  # 像素值转到[0~255]

                        if band == 1:
                            data = dataraster.reshape((srcYSize, srcXSize, 1))
                        else:
                            # 将每个波段的数组很并到一个3s维数组中
                            data = np.append(data, dataraster.reshape((srcYSize, srcXSize, 1)), axis=2)
                        del dataraster, srcband
                    srcDS = None
                    # data_1 = data * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
                    data_1 = data
                    data = None
                    # print(data_1.shape)

                    flagY = True
                    for i in range(0, srcYSize, isize):
                        if not flagY:
                            break
                        if i + isize > srcYSize:
                            i_end = srcYSize
                            flagY = False
                        else:
                            i_end = i + isize

                        flagX = True
                        for j in range(0, srcXSize, isize):
                            if not flagX:
                                break
                            if j + isize > srcXSize:
                                j_end = srcXSize
                                flagX = False
                            else:
                                j_end = j + isize
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

                            dataraster = (data_1[i:i_end, j:j_end])
                            dataraster = dataraster * (1. / 255) - 0.5
                            dataraster_new = np.lib.pad(dataraster,
                                                        (
                                                            (0, isize - (i_end - i)), (0, isize - (j_end - j)), (0, 0)),
                                                        'constant', constant_values=0)
                            dataraster_new = np.array(dataraster_new)

                            data = dataraster_new.reshape([-1, img_size])
                            ori_y = i
                            ori_x = j

                            pred1 = sess.run(pred_new, feed_dict={x: data, keep_prob: 1.})
                            value_1 = np.argmax(pred1, axis=1)

                            # 重新生成掩膜图像，2017.10.11,by xjxf __start
                            mask_value = value_1.reshape([isize, isize])
                            # 重新生成掩膜图像，2017.10.11,by xjxf __end

                            # 将块写入掩膜图像，2017.10.11,by xjxf __start
                            data_target_temp[ori_y:i_end, ori_x:j_end] = mask_value[0:i_end - ori_y,
                                                                         0:j_end - ori_x]
                            # 将块写入掩膜图像，2017.10.11,by xjxf __end
                            # print("ori_x:",str(ori_x),"ori_y",str(ori_y))
                            # target_ds.FlushCache()  # 将数据写入文件

                    if task_index == 0:
                        # raster_fn = path.join(dir_name, 'test_mask.tiff')  # 存放掩膜影像
                        # raster_fn = dir_name + "_mask.tif"
                        raster_fn = FLAGS.save_image_path
                        if not os.path.exists(raster_fn): os.makedirs(raster_fn)
                        # raster_fn += dir_name.split('/')[-1].split('.')[0] + "_mask.tif"

                        raster_fn = os.path.join(raster_fn,dir_name.split('/')[-1].split('.')[0] + "_mask.tif")

                        # raster_fn ="test_mask_1.tiff"

                        target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, srcXSize, srcYSize, 1,
                                                                         gdal.GDT_Byte)
                        target_ds.SetGeoTransform(geoTrans)  # 设置掩膜的地理参考
                        target_ds.SetProjection(srcPro)  # 设置掩膜坐标引用
                        target_ds.GetRasterBand(1).SetNoDataValue(0)

                        del geoTrans
                        del srcPro
                        del dir_name, raster_fn
                        target_ds.GetRasterBand(1).WriteArray(data_target_temp, 0, 0)
                        target_ds.FlushCache()  # 将数据写入文件
                        target_ds = None

                    print("推理完成：100.00%")
                    # endtime = datetime.datetime.now()
                    # print('end time:', datetime.datetime.now())
                    # print((endtime - starttime).seconds)

            def Inference2(task_index):
                img_paths = []
                # 读gtdata
                # [img_paths.append('gtdata:////users/xiaoshaolin/tensorflow/hubei/' + i.replace('\n', ''))
                #  for i in open('./tile.log')]  # 从hdfs上读取文件tile.log
                # 读hdfs
                [img_paths.append('HDFS://dm01-08-01.tjidc.dcos.com:8020/user/root/hubei_title/' + i.replace('\n', ''))
                 for i in open('./tile.log')]  # 从hdfs上读取文件tile.log

                # isize = img_pixel  # 10 x 10的样本
                m = 1
                isizes = m * isize
                # img_size = img_pixel * img_pixel * channels

                # 为了支持中文路径，请添加下面这句代码
                gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")

                gdal.AllRegister()  # 注册驱动
                ogr.RegisterAll()

                start = datetime.datetime.now()
                print('session start:', datetime.datetime.now())
                end_index = min(FLAGS.num_images + FLAGS.start_index, len(img_paths))
                for dir_name in img_paths[FLAGS.start_index:end_index]:
                    srcDS = gdal.Open(dir_name, GA_ReadOnly)  # 只读方式打开原始影像

                    geoTrans = srcDS.GetGeoTransform()  # 获取地理参考6参数
                    srcPro = srcDS.GetProjection()  # 获取坐标引用
                    srcXSize = srcDS.RasterXSize  # 宽度
                    srcYSize = srcDS.RasterYSize  # 高度
                    nbands = srcDS.RasterCount  # 波段数

                    dBuf_1 = np.zeros([srcYSize, srcXSize], np.uint8)  # 整个掩膜的缓存

                    # saver = tf.train.Saver()  # 默认是保存所有变量
                    # cdef int i,j,band,xx,yy


                    # ckpt = tf.train.get_checkpoint_state(logdir)
                    # if ckpt and ckpt.model_checkpoint_path:
                    #     saver.restore(sess, ckpt.model_checkpoint_path)

                    # 一次性读取图像所有数据
                    for band in range(nbands):
                        # band=int(band)
                        band += 1
                        # print "[ GETTING BAND ]: ", band
                        srcband = srcDS.GetRasterBand(band)  # 获取该波段
                        if srcband is None:
                            continue

                        # Read raster as arrays 类似RasterIO（C++）
                        dataraster = srcband.ReadAsArray(0, 0, srcXSize, srcYSize, srcXSize,
                                                         srcYSize)  # .astype(np.uint8)  # 像素值转到[0~255]

                        if band == 1:
                            data = dataraster.reshape((srcYSize, srcXSize, 1))
                        else:
                            # 将每个波段的数组很并到一个3s维数组中
                            data = np.append(data, dataraster.reshape((srcYSize, srcXSize, 1)), axis=2)
                        del dataraster, srcband
                    srcDS = None
                    # data_1 = data * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
                    data_1 = data

                    def compute(YSize1, YSize2):  # 每次计算YSize1~YSize2 行，实现伪并行计算
                        flagY = True

                        for i in range(YSize1, YSize2, isizes):
                            # print("正在计算：%.2f%%" % (i/srcYSize * 100), end="\r")
                            if not flagY: break
                            if i + isizes > YSize2 - 1:
                                i = YSize2 - 1 - isizes
                                flagY = False

                            # print('loop start:', datetime.datetime.now())

                            flagX = True
                            for j in range(0, srcXSize, isizes):
                                if not flagX: break

                                if j + isizes > srcXSize - 1:
                                    j = srcXSize - 1 - isizes
                                    flagX = False

                                data2 = data_1[i:i + isize, j:j + isize].reshape([-1, img_size])

                                if j == 0:
                                    data_2_1 = data2
                                else:
                                    data_2_1 = np.vstack((data_2_1, data2))  # 取出一个isize行所有的image,再进行预测（而不是一张张预测）

                            # 标准化
                            # Z-score标准化方法
                            # mean = np.reshape(np.average(data_2_1, 1), [np.shape(data_2_1)[0], 1])
                            # std = np.reshape(np.std(data_2_1, 1), [np.shape(data_2_1)[0], 1])
                            #
                            # data_2_1 = (data_2_1 - mean) / std

                            # max_ = np.reshape(np.max(data_2_1, 1), [np.shape(data_2_1)[0], 1])
                            # min_ = np.reshape(np.min(data_2_1, 1), [np.shape(data_2_1)[0], 1])
                            # data_2_1 = (data_2_1 - min_) / (max_ - min_)

                            data_2_1 = data_2_1 * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5

                            # """
                            # 10x10像素预测

                            for row, pred1 in enumerate(sess.run(pred, feed_dict={x: data_2_1, keep_prob: 1.})):
                                if np.argmax(pred1):
                                    # print('pred3', pred3)
                                    jj = row * isize
                                    if jj + isize >= srcXSize - 1:
                                        jj = srcXSize - 1 - isize
                                    dBuf_1[i:i + isize, jj:jj + isize] = np.ones([isize, isize], np.uint8) * 255
                            # """

                    compute(0, srcYSize)
                    print(datetime.datetime.now() - start)
                    print('writer data:', datetime.datetime.now())

                    if task_index == 0:
                        # raster_fn = path.join(dir_name, 'test_mask.tiff')  # 存放掩膜影像
                        # raster_fn = dir_name + "_mask.tif"
                        raster_fn = FLAGS.save_image_path
                        if not os.path.exists(raster_fn): os.makedirs(raster_fn)
                        raster_fn += dir_name.split('/')[-1].split('.')[0] + "_mask.tif"
                        # raster_fn ="test_mask_1.tiff"

                        target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, srcXSize, srcYSize, 1,
                                                                         gdal.GDT_Byte)
                        target_ds.SetGeoTransform(geoTrans)  # 设置掩膜的地理参考
                        target_ds.SetProjection(srcPro)  # 设置掩膜坐标引用
                        # band = target_ds.GetRasterBand(1)  # 获取第一个波段(影像只有一个波段)
                        # band.SetNoDataValue(0)  # 将这个波段的值全设置为0
                        target_ds.GetRasterBand(1).SetNoDataValue(0)

                        del geoTrans
                        del srcPro
                        del dir_name, raster_fn
                        target_ds.GetRasterBand(1).WriteArray(dBuf_1, 0, 0)
                        target_ds.FlushCache()  # 将数据写入文件
                        target_ds = None

            while not sv.should_stop():
                if FLAGS.model == 1:
                    Inference(FLAGS.task_index)
                if FLAGS.model == 2:
                    Inference2(FLAGS.task_index)
                # Inference2(0)
                break
            end_time = datetime.datetime.now()
            print("总时间: ", end_time - start_time)
            sess.close()
            exit()

if __name__ == "__main__":
    tf.app.run()
