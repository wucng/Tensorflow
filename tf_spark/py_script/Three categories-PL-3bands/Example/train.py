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
import cv2

bid_dict={
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

# 训练集文件路径
dir_name = ''
dir_summary_name=''

#生成模型最终会保存在model文件夹下
model_save_path = "model/"

# # 输出文件路径设置
fpa_path = path.join(dir_name, 'train_output.txt')
fpa = open(fpa_path, "a")      #这个文件好像没什么用 by xjxf
# # fpa.close()

start_time = datetime.datetime.now()
print("startTime: ", start_time)

# 提起pickle数据 data包含 特征+标签
data = tool_set.read_and_decode(dir_name + "train_data_400_all.pkl",3)

isize =400
img_channel =3

img_pixel = isize

'''
# CNN 完整程序  训练模型
'''
# Parameters
training_epochs =2500
batch_size = 1
display_step = 1
channels = img_channel

# Network Parameters
img_size = isize * isize * channels  # data input (img shape: 28*28*3)
label_cols =3 # total classes (云、非云)使用标签[1,0,0] 3维             #问题1，labels_cols是干什么用的？  by xjxf
dropout = 0.8 # Dropout, probability to keep units                        #问题2，dropout为什么设为0.75    by xjxf
img_nums = data.shape[0]
img_pixel_h=isize
img_pixel_w=isize


print(img_nums)

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
# decay_steps=3curl command
# decay_steps = int((num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)/2)
decay_steps = int(num_batches_per_epoch*2)
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
    # Reshape input picture  x shape [batch,400*400*3]
    x = tf.reshape(x, shape=[-1, img_pixel_h, img_pixel_w, channels])  # [batch,400,400,3]

    # Convolution Layer
    conv1_1,relu1_1= conv_relu(x, 'wc1_1')  # [batch,400,400,64]
    conv1_2,relu1_2=conv_relu(relu1_1,'wc1_2') # [batch,400,400,64]
    pool_1=max_pool(relu1_2) # [batch,200,200,64]

    conv2_1, relu2_1 = conv_relu(pool_1, 'wc2_1')  # [batch,200,200,128]
    conv2_2, relu2_2 = conv_relu(relu2_1, 'wc2_2') # [batch,200,200,128]
    pool_2 = max_pool(relu2_2) # [batch,100,100,128]

    conv3_1, relu3_1 = conv_relu(pool_2, 'wc3_1')  # [batch,100,100,256]
    conv3_2, relu3_2 = conv_relu(relu3_1, 'wc3_2') # [batch,100,100,256]
    conv3_3, relu3_3 = conv_relu(relu3_2, 'wc3_3') # [batch,100,100,256]
    pool_3 = max_pool(relu3_3) # [batch,50,50,256]

    conv4_1, relu4_1 = conv_relu(pool_3,'wc4_1')  # # [batch,50,50,512]
    conv4_2, relu4_2 = conv_relu(relu4_1, 'wc4_2') # [batch,50,50,512]
    conv4_3, relu4_3 = conv_relu(relu4_2, 'wc4_3') # [batch,50,50,512]
    pool_4 = max_pool(relu4_3) # [batch,25,25,512]

    conv5_1, relu5_1 = conv_relu(pool_4, 'wc5_1')  # [batch,25,25,512]
    conv5_2, relu5_2 = conv_relu(relu5_1, 'wc5_2') # [batch,25,25,512]
    conv5_3, relu5_3 = conv_relu(relu5_2, 'wc5_3') # [batch,25,25,512]
    pool_5 = max_pool(relu5_3) # # [batch,13,13,512]

    fc_6,relu_fc6 = conv_relu(pool_5,'wf_6') # [batch,13,13,4096]
    relu_fc6=tf.nn.dropout(relu_fc6,dropout) # [batch,13,13,4096]

    fc_7,relu_fc7=conv_relu(relu_fc6,'wf_7') # [batch,13,13,4096]
    relu_fc7=tf.nn.dropout(relu_fc7,dropout) # [batch,13,13,4096]

    out_1,relu_out1=conv_relu(relu_fc7,'w_out') # [batch,13,13,3]


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
                                 strides=[1,2,2,1],padding="SAME") # [batch_size,img_pixel_h//16,img_pixel_w//16,label_cols] 即[batch,25,25,3]
    # 上采样至pool_4的大小  2017.10.13,by xjxf   __end

    #让pool_4卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start
    upsample_pool4,upsample_pool4_relu=conv_relu(pool_4,'up_sample_p_4') # [batch,25,25,3]
    out_upsample_pool4=upsample_pool4+out_end_1 # [batch,25,25,3]
    out_end_2=tf.nn.conv2d_transpose(out_upsample_pool4,tf.get_variable("up_sample_4",shape=[4,4,label_cols,label_cols],
                                                                        initializer=tf.glorot_uniform_initializer()),
                                     output_shape=[batch_size,img_pixel_h//8,img_pixel_w//8,label_cols],
                                     strides=[1,2,2,1],padding='SAME') # [batch_size,img_pixel_h//8,img_pixel_w//8,label_cols] 即 [batch,50,50,3]
    # 让pool_4卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start

    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start
    upsample_pool3, upsample_pool3_relu = conv_relu(pool_3, 'up_sample_p_3') # [batch,50,50,3]
    out_upsample_pool3 = upsample_pool3 + out_end_2 # [batch,50,50,3]
    out_end_3 = tf.nn.conv2d_transpose(out_upsample_pool3,
                                       tf.get_variable("up_sample_3", shape=[4, 4, label_cols, label_cols],
                                                       initializer=tf.glorot_uniform_initializer()),
                                       output_shape=[batch_size, img_pixel_h//4, img_pixel_w//4, label_cols],
                                       strides=[1, 2, 2, 1], padding='SAME') # [batch_size, img_pixel_h//4, img_pixel_w//4, label_cols] 即 [batch,100,100,3]
    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.12,by xjxf  __start

    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.16,by xjxf  __start
    upsample_pool2, upsample_pool2_relu = conv_relu(pool_2, 'up_sample_p_2') # [batch,100,100,3]
    out_upsample_pool2 = upsample_pool2 + out_end_3 # [batch,100,100,3]
    out_end_4 = tf.nn.conv2d_transpose(out_upsample_pool2,
                                       tf.get_variable("up_sample_2", shape=[8, 8, label_cols, label_cols],
                                                       initializer=tf.glorot_uniform_initializer()),
                                       output_shape=[batch_size, img_pixel_h, img_pixel_w, label_cols],
                                       strides=[1, 4, 4, 1], padding='SAME') # [batch_size, img_pixel_h, img_pixel_w, label_cols] 即 [batch,400,400,3]
    # 让pool_3卷积输出为第一维为labcolS大小，并和out_end_1做融合，2017.10.16,by xjxf  __start
    return out_end_4 # [batch,400,400,3]

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


# 初始化所有的op
init = tf.global_variables_initializer()

if __name__ == '__main__':
    # print("xjxf:"+str(img_nums))             #for test by xjxf
    saver = tf.train.Saver()  # 默认是保存所有变量
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)  # 创建输出文件目录
    model_fn = path.join(model_save_path, 'save_net.ckpt')  # 存放掩膜影像
    l_r_1=0
    count=0
    initial_count=tf.constant(0)
    update_count=tf.assign(global_step,initial_count)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:   #这句是什么意思？

        # 合并到Summary中,并选定可视化存储目录,2017.10.11,by xjxf __start
        merged = tf.summary.merge_all()
        write = tf.summary.FileWriter(dir_summary_name, sess.graph)
        # 合并到Summary中,并选定可视化存储目录2017.10.11,by xjxf __end

        sess.run(init)
        # saver.restore(sess,model_fn)
        sess.run(update_count)
        total_batch = int(img_nums / batch_size)

        for epoch in range(training_epochs):

            # index = [i for i in range(len(data))]
            # np.random.shuffle(index)            #只会打乱第一维度的顺序
            # data = data[index]
            np.random.shuffle(data)

            flag=1
            avg_cost = 0.
            for i in range(total_batch):
                img, label = tool_set.next_batch(data, batch_size, flag,img_pixel=isize, channels=img_channel)
                flag=0
                batch_xs = img.reshape([-1, img_size])
                batch_ys = tool_set.dense_to_one_hot(label[:, np.newaxis], label_cols)  # 生成多列标签   问题6，生成多列标签是干什么呢？   by xjxf
                # global_step+=1
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_xs,
                                                              y: batch_ys, keep_prob: dropout})
                count+=1
                # print(count)
                learning_rate_1=sess.run(learning_rate)
                if l_r_1 !=learning_rate_1:
                    l_r_1=learning_rate_1
                    print("learning_rate:",str(learning_rate_1))
                # global_step_1=sess.run(global_step)
                # print(global_step_1)
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("time: ", datetime.datetime.now())
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost:", "{:.9f}".format(avg_cost),
                      'accuracy:', sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0}))
                result,pred_result=sess.run([merged,pred_new],feed_dict={x: batch_xs,y: batch_ys,keep_prob:1.0})
                # pred_result=np.argmax(pred_result,axis=1)
                # pred_result_1=pred_result.reshape([isize,isize])*255
                # cv2.imwrite("pre_result"+str(epoch)+'.tif',pred_result_1)
                write.add_summary(result,epoch)
                save_path = saver.save(sess, model_fn)  # 保留训练的模型

        print("Optimization Finished!")


        print('Saver path:', save_path)

        fpa.close()

        end_time = datetime.datetime.now()
        print("end_time: ", end_time)
        print("time used: ", end_time-start_time)
