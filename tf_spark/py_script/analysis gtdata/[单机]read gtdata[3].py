# -*- coding:utf-8 -*-

'''
--------------单机版----------------
直接从gtdata上读取图像与对应的掩膜图像，生成图像数据+标签数据 卷积模型
执行命令：
spark-submit
--queue default \
--num-executors 45 \
--executor-memory 2G \
--driver-memory 12G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--archives hdfs://xxx:8020/user/root/mnist/mnist.zip#mnist \
xxx.py
或
python mnist_dist.py 
或
spark-submit mnist_dist.py 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
try:
  from osgeo import gdal
except:
  import gdal

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

def create_pickle_train(image_path, mask_path, img_pixel=2, channels=3):
    # m = 0
    # image_data = Multiband2Array(image_path)
    image_data=Multiband2Array(image_path)
    # mask_data = cv2.split(cv2.imread(mask_path))[0] / 255
    # mask_data=np.asarray(Image.open(mask_path))/255

    mask_data=Multiband2Array(mask_path)/255

    x_size, y_size = image_data.shape[:2]

    data_list = []
    m=0
    for i in range(0, x_size - img_pixel + 1, img_pixel // 2):  # 文件夹下的文件名
        if i + img_pixel > x_size:
            i = x_size - img_pixel - 1
        for j in range(0, y_size - img_pixel + 1, img_pixel // 2):
            if j + img_pixel > y_size:
                j = y_size - img_pixel - 1
            cropped_data = image_data[i:i + img_pixel, j:j + img_pixel]
            data1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))  # 展成一行
            train_label = mask_data[i:i + img_pixel, j:j + img_pixel].max()
            # train_label = 1
            # train_label = mask_data[i:i + img_pixel, j:j + img_pixel].min()
            # train_label = int(mask_data[i:i + img_pixel, j:j + img_pixel].sum() / (img_pixel*img_pixel/2+1))
            data2 = np.append(data1, train_label)[np.newaxis, :]  # 数据+标签
            data_list.append(data2)
            m+=1
            #'''
            if m % 1000 == 0:
                # print(datetime.datetime.now(), "compressed {number} images".format(number=m))
                data_matrix = np.array(data_list, dtype=np.float32)
                data_matrix = data_matrix.reshape((-1, img_pixel * img_pixel * channels + 1))
                return data_matrix
            #'''
    data_matrix = np.array(data_list, dtype=np.float32)
    data_matrix = data_matrix.reshape((-1, img_pixel*img_pixel*channels+1))
    return data_matrix

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  # 从标量类标签转换为一个one-hot向量
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  # print index_offset
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def dense_to_one_hot2(labels_dense,num_classes):
    labels_dense=np.array(labels_dense,dtype=np.uint8)
    num_labels = labels_dense.shape[0] # 标签个数
    labels_one_hot=np.zeros((num_labels,num_classes),np.float32)
    for i,itenm in enumerate(labels_dense):
        labels_one_hot[i,itenm]=1
    return labels_one_hot


IMAGE_PIXELS=2 # 图像大小 mnist 28x28x1  (后续参考自己图像大小进行修改)
channels=3
num_class=2
# batch_size = args.batch_size
dropout = 0.7
INITIAL_LEARNING_RATE=0.01


image_path='gtdata:///users/xiaoshaolin/tensorflow/GF/PART2/11.tif'

mask_path='gtdata:///users/xiaoshaolin/tensorflow/GF/PART2/11_mask.tif'

data=create_pickle_train(image_path,mask_path)

print(data.shape)
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
    img = img.astype(np.float32)

    img = (img - np.min(img, 0)) / (np.max(img, 0) - np.min(img, 0)+0.001)

    label = data1[:, -1:]
    label = label.astype(np.float32)  # 类型转换

    return img,label


def main(_):


        # Create the model
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
            'wc1': tf.get_variable('wc1', [3, 3, channels, 128], dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
        # 5X5的卷积模板

            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.get_variable('wc2', [3, 3, 32, 64], dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),

            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([(IMAGE_PIXELS // 2) * (IMAGE_PIXELS // 2) * 128, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, num_class]))
        }

        biases = {
            'bc1': tf.get_variable('bc1', [128], dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
            'bc2': tf.get_variable('bc2', [64], dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([num_class]))
        }

        # Placeholders or QueueRunner/Readers for input data
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS * channels], name="x")  # mnist 28*28*1
        y_ = tf.placeholder(tf.float32, [None, num_class], name="y_")
        keep = tf.placeholder(tf.float32)
        # is_training = tf.placeholder(tf.bool, name='MODE')

        x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, channels])  # mnist 数据 28x28x1 (灰度图 波段为1)

        # x_img=batch_norm_layer(x_img,is_training)
        x_img = tf.nn.lrn(x_img, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)  # lrn层

        # 改成卷积模型
        conv1 = conv2d(x_img, weights['wc1'], biases['bc1'])
        conv1 = maxpool2d(conv1, k=2)  # shape [N,1,1,32]
        conv1 = tf.nn.lrn(conv1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)  # lrn层
        # conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # conv2 = maxpool2d(conv2, k=2)  # shape [N,1,1,32]
        # conv1 = tf.nn.dropout(conv1, keep+0.1)
        fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        # fc1=batch_norm_layer(fc1, is_training)
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep)
        y = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        # y=tf.sigmoid(y) # 二分类 多分类加 tf.nn.softmax()

        global_step = tf.Variable(0, name="global_step", trainable=False)

        # loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        # learning_rate=tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,
        #                                          learning_rate_decay_steps,learning_rate_decay_rate,
        #                                          staircase=False)

        # learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
        #                                            global_step,
        #                                            10000,
        #                                            0.96,
        #                                            staircase=False)
        learning_rate = tf.train.polynomial_decay(INITIAL_LEARNING_RATE, global_step, 3000000, 1e-5, 0.8, False)
        # 运行steps：decay_steps>1000:1
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
        summary_op = tf.summary.merge_all()

        init_op = tf.global_variables_initializer()

        logdir="./model_arable2"
        # Create a "Supervisor", which oversees the training process.
        # sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
        #                          logdir=logdir,
        #                          init_op=init_op,
        #                          # summary_op=None,
        #                          saver=saver,
        #                          # saver=None, # None 不自动保存模型
        #                          # recovery_wait_secs=1,
        #                          global_step=global_step,
        #                          stop_grace_secs=300,
        #                          save_model_secs=10)

        sess=tf.InteractiveSession()
        sess.run(init_op)

        # 验证之前是否已经保存了检查点文件
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # The supervisor takes care of session initialization and restoring from
        # a checkpoint.
        # sess = sv.prepare_or_wait_for_session(server.target)

        # Start queue runners for the input pipelines (if ang).
        # sv.start_queue_runners(sess)

        # Loop until the supervisor shuts down (or 2000 steps have completed).
        step = 0
        while step < 2000:
            # batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_xs, batch_ys = next_batch(train_data, 100)
            batch_ys=dense_to_one_hot2(batch_ys,num_class)
            # _, step = sess.run([train_op, global_step], feed_dict={x: batch_xs, y_: batch_ys,keep:dropout})
            sess.run([train_op], feed_dict={x: batch_xs, y_: batch_ys, keep: dropout})
            if step % 100 == 0:
                print("accuracy: %f" % sess.run(accuracy, feed_dict={x: test_data[:,:-1],
                                                                     y_: dense_to_one_hot2(test_data[:,-1],num_class),keep:1.0}))
            step+=1

if __name__ == "__main__":
    tf.app.run()
