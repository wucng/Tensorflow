# -*- coding:utf-8 -*-
# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

# Distributed MNIST on grid based on TensorFlow MNIST example
"""
可以实现保存精度比较高的模型参数

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import nested_scopes
from __future__ import print_function


def print_log(worker_num, arg): # 打印信息
  print("{0}: {1}".format(worker_num, arg))
  # log.info("{0}: {1}".format(worker_num, arg))

def map_fun(args, ctx):
  # from com.yahoo.ml.tf import TFNode
  from tensorflowonspark import TFNode
  from datetime import datetime
  import math
  import numpy as np
  import tensorflow as tf
  from tensorflow.contrib.layers.python.layers import batch_norm
  import time
  import os

  worker_num = ctx.worker_num #worker数量
  job_name = ctx.job_name # job名
  task_index = ctx.task_index # 任务索引
  cluster_spec = ctx.cluster_spec # 集群

  IMAGE_PIXELS=16 # 图像大小 mnist 28x28x1  (后续参考自己图像大小进行修改)
  channels=3
  num_class=IMAGE_PIXELS*IMAGE_PIXELS
  # global dropout
  dropout = args.dropout
  # Parameters
  # hidden_units = 128 # NN隐藏层
  # training_epochs=args.epochs
  batch_size   = args.batch_size #每批次训练的样本数
  # img_nums=630000
  # global learning_rate
  # learning_rate=args.learning_rate
  INITIAL_LEARNING_RATE=args.learning_rate
  # flag=True

  # batch_size=200

  num_examples_per_epoch_for_train = (4015 - 1) ** 2 # 每次迭代的样本数
  num_batches_per_epoch=int(num_examples_per_epoch_for_train/batch_size)
  num_epochs_per_decay=1.2
  learning_rate_decay_rate=0.8
  learning_rate_decay_steps=int(num_batches_per_epoch*num_epochs_per_decay)

  """
  # ---------设置动态学习效率
  # Constants describing the training process.
  # MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
  NUM_EPOCHS_PER_DECAY = batch_size  # Epochs after which learning rate decays.
  LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
  INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

  global_step1 = training_epochs * (img_nums // batch_size)  # Integer Variable counting the number of training steps
  # Variables that affect learning rate.
  num_batches_per_epoch = img_nums / batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                             global_step1,
                                             decay_steps,
                                             LEARNING_RATE_DECAY_FACTOR,
                                             staircase=True)
# 设置动态学习效率----------  
"""


  # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
  if job_name == "ps": # ps节点(主节点)
    time.sleep((worker_num + 1) * 5)


  # Get TF cluster and server instances
  cluster, server = TFNode.start_cluster_server(ctx, 1, args.rdma)

  def feed_dict(batch):
    # Convert from [(images, labels)] to two numpy arrays of the proper type
    images = []
    labels = []
    np.random.shuffle(batch) # 随机打乱
    for item in batch:
      images.append(item[0])
      labels.append(item[1])
    xs = np.array(images)
    xs = xs.astype(np.float32)
    # xs = xs/255.0 # 数据归一化
    # Z-score标准化方法
    # mean = np.reshape(np.average(xs, 1), [np.shape(xs)[0], 1])
    # std = np.reshape(np.std(xs, 1), [np.shape(xs)[0], 1])
    # xs = (xs - mean) / std

    # min-max标准化（Min-Max Normalization
    max_ = np.reshape(np.max(xs, 1), [np.shape(xs)[0], 1])
    min_ = np.reshape(np.min(xs, 1), [np.shape(xs)[0], 1])

    xs = (xs - min_) / (max_ - min_+0.0001)
    ys = np.array(labels)
    ys=np.reshape(ys,[-1,IMAGE_PIXELS,IMAGE_PIXELS])
    ys = ys.astype(np.uint8)
    return (xs, ys)

  def batch_norm_layer(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: batch_norm(inputT, is_training=True,
                                      center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                   lambda: batch_norm(inputT, is_training=False,
                                      center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                                      scope=scope))  # , reuse = True))

  if job_name == "ps":
    server.join()
  elif job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % task_index,
        cluster=cluster)):

      # Create some wrappers for simplicity
      def conv2d(x, W, b, is_training=True,strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)  # strides中间两个为1 表示x,y方向都不间隔取样
        x = batch_norm_layer(x, is_training)
        return tf.nn.relu(x)

      def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')  # strides中间两个为2 表示x,y方向都间隔1个取样

      # Store layers weight & bias
      weights = {
        'wc1': tf.get_variable('wc1', [3, 4, channels, 64], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
        # 'wc2': tf.get_variable('wc2',[3,3,64,64],dtype=tf.float32,
        #                        initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss)*0.001,

        'wc3': tf.get_variable('wc3', [3, 3, 64, 128], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
        # 'wc4': tf.get_variable('wc4', [3, 3, 128, 128], dtype=tf.float32,
        #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,

        'wc5': tf.get_variable('wc5', [3, 3, 128, 256], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
        'wc6': tf.get_variable('wc6', [3, 3, 256, 256], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
        'wc7': tf.get_variable('wc7', [3, 3, 256, 256], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
        # 'wc8': tf.get_variable('wc8', [3, 3, 256, 256], dtype=tf.float32,
        #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,

        # 'wc9': tf.get_variable('wc9', [3, 3, 256, 256], dtype=tf.float32,
        #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
        # 'wc10': tf.get_variable('wc10', [3, 3, 256, 256], dtype=tf.float32,
        #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
        # 'wc11': tf.get_variable('wc11', [3, 3, 256, 256], dtype=tf.float32,
        #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
        # 'wc12': tf.get_variable('wc12', [3, 3, 256, 256], dtype=tf.float32,
        #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss) * 0.001,
      }

      biases = {
        'bc1': tf.get_variable('bc1', [64], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
        'bc2': tf.get_variable('bc2', [64], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),

        'bc3': tf.get_variable('bc3', [128], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
        'bc4': tf.get_variable('bc4', [128], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),

        'bc5': tf.get_variable('bc5', [256], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
        'bc6': tf.get_variable('bc6', [256], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
        'bc7': tf.get_variable('bc7', [256], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
        'bc8': tf.get_variable('bc8', [256], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),

        # 'bc9': tf.get_variable('bc9',[256],dtype=tf.float32,
        #                        initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
        # 'bc10': tf.get_variable('bc10',[256],dtype=tf.float32,
        #                        initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
        # 'bc11': tf.get_variable('bc11', [256], dtype=tf.float32,
        #                        initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),
        # 'bc12': tf.get_variable('bc12',[256],dtype=tf.float32,
        #                        initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
      }

      # Placeholders or QueueRunner/Readers for input data
      x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS * channels], name="x")  # mnist 28*28*1
      y_ = tf.placeholder(tf.float32, [None, IMAGE_PIXELS,IMAGE_PIXELS], name="y_")
      keep = tf.placeholder(tf.float32)
      is_training = tf.placeholder(tf.bool, name='MODE')

      x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, channels])  # mnist 数据 28x28x1 (灰度图 波段为1)
      # tf.summary.image("x_img", x_img)

      # 改成卷积模型
      conv1 = conv2d(x_img, weights['wc1'], biases['bc1'], is_training)
      # conv1=conv2d(conv1,weights['wc2'],biases['bc2'],is_training)
      conv1 = tf.nn.lrn(conv1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
      conv1 = maxpool2d(conv1, k=2)
      conv1 = tf.nn.dropout(conv1, keep)

      conv2 = conv2d(conv1, weights['wc3'], biases['bc3'], is_training)
      # conv2 = conv2d(conv2, weights['wc4'], biases['bc4'],is_training)
      conv2 = tf.nn.lrn(conv2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
      conv2 = maxpool2d(conv2, k=2)
      conv2 = tf.nn.dropout(conv2, keep)

      conv3 = conv2d(conv2, weights['wc5'], biases['bc5'], is_training)
      # conv3 = conv2d(conv3, weights['wc6'], biases['bc6'],is_training)
      conv3 = tf.nn.lrn(conv3, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
      conv3 = maxpool2d(conv3, k=2)
      conv3 = tf.nn.dropout(conv3, keep)

      conv4 = conv2d(conv3, weights['wc7'], biases['bc7'], is_training)
      # conv4 = conv2d(conv4, weights['wc8'], biases['bc8'],is_training)
      conv4 = tf.nn.lrn(conv4, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
      conv4 = maxpool2d(conv4, k=2)

      # conv4 = conv2d(conv4, weights['wc9'], biases['bc9'],is_training)
      # conv4 = conv2d(conv4, weights['wc10'], biases['bc10'],is_training)
      # conv4 = conv2d(conv4, weights['wc11'], biases['bc11'],is_training)
      # conv4 = conv2d(conv4, weights['wc12'], biases['bc12'],is_training)
      # conv4=tf.nn.lrn(conv4, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
      # conv4 = maxpool2d2(conv4, k=2)
      y = tf.reshape(conv4, [-1, num_class])

      global_step = tf.Variable(0, name="global_step", trainable=False)

      # loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

      # loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

      y = tf.reshape(y, [-1, IMAGE_PIXELS, IMAGE_PIXELS])
      # y_ = tf.reshape(y_, [-1, IMAGE_PIXELS, IMAGE_PIXELS])
      # loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)))

      # loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)))
      loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y))
      # learning_rate=tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,
      #                                          learning_rate_decay_steps,learning_rate_decay_rate,
      #                                          staircase=False)

      # learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
      #                                            global_step,
      #                                            10000,
      #                                            0.96,
      #                                            staircase=False)
      learning_rate=tf.train.polynomial_decay(INITIAL_LEARNING_RATE,global_step,3000000,1e-5,0.8,False)
      # 运行steps：decay_steps>1000:1
      # train_op = tf.train.AdagradOptimizer(learning_rate).minimize(
      #     loss, global_step=global_step)

      train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)

      # Test trained model
      # label = tf.argmax(y_, 1, name="label")
      # prediction = tf.argmax(y, 1,name="prediction")
      # correct_prediction = tf.equal(prediction, label)
      #
      # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
      # tf.summary.scalar("acc", accuracy)

      # def compute_acc(xs, ys, IMAGE_PIXELS):
      #   global y
      #   y1 = sess.run(y, {x: xs, y: ys, keep: 1., is_training: True})
      #   prediction = [1. if abs(x2 - 1) < abs(x2 - 0) else 0. for x1 in y1 for x2 in x1]
      #   prediction = np.reshape(prediction, [-1, IMAGE_PIXELS, IMAGE_PIXELS]).astype(np.uint8)
      #   # correct_prediction = np.equal(prediction, np.reshape(ys,[-1, IMAGE_PIXELS, IMAGE_PIXELS])).astype(tf.float32)
      #   # accuracy = np.mean(correct_prediction)
      #   accuracy = np.mean(np.equal(prediction, np.reshape(ys, [-1, IMAGE_PIXELS, IMAGE_PIXELS])).astype(np.float32))
      #   return accuracy
      # def compute_acc(xs, ys, IMAGE_PIXELS):
      #   global y
      #   y1 = sess.run(y, {x: xs, y_: ys, keep: 1., is_training: True})
      #   prediction = [1. if abs(x3 - 1) < abs(x3 - 0) else 0. for x1 in y1 for x2 in x1 for x3 in x2]
      #   prediction = np.reshape(prediction, [-1, IMAGE_PIXELS, IMAGE_PIXELS]).astype(np.uint8)
      #   # correct_prediction = np.equal(prediction, np.reshape(ys,[-1, IMAGE_PIXELS, IMAGE_PIXELS])).astype(tf.float32)
      #   # accuracy = np.mean(correct_prediction)
      #   accuracy = np.mean(np.equal(prediction, ys).astype(np.float32))
      #   return accuracy

      saver = tf.train.Saver()

      # summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()


    # Create a "supervisor", which oversees the training process and stores model state into HDFS
    logdir = TFNode.hdfs_path(ctx, args.model)
    print("tensorflow model path: {0}".format(logdir)) #
    # log.info("tensorflow model path: {0}".format(logdir))
    # summary_writer = tf.summary.FileWriter("tensorboard_%d" %(worker_num), graph=tf.get_default_graph())


    if args.mode == "train":
      sv = tf.train.Supervisor(is_chief=(task_index == 0),
                               logdir=logdir,
                               init_op=init_op,
                               # summary_op=None,
                               saver=saver,
                               # saver=None, # None 不自动保存模型
                               # recovery_wait_secs=1,
                               global_step=global_step,
                               stop_grace_secs=300,
                               save_model_secs=10)
    elif args.mode == "retrain":
      sv = tf.train.Supervisor(is_chief=(task_index == 0),
                               logdir=logdir,
                               # init_op=init_op,
                               # summary_op=None,
                               # saver=None, # None 不自动保存模型
                               saver=saver,
                               # recovery_wait_secs=1,
                               global_step=global_step,
                               stop_grace_secs=300,
                               save_model_secs=10)
    else:
      sv = tf.train.Supervisor(is_chief=(task_index == 0),
                               logdir=logdir,
                               # summary_op=None,
                               saver=saver,
                               # recovery_wait_secs=1,
                               global_step=global_step,
                               stop_grace_secs=300,
                               save_model_secs=0)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess: # 打开session

      """
      # 验证之前是否已经保存了检查点文件
      ckpt = tf.train.get_checkpoint_state(logdir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
      """
        # global_step=int(ckpt.model_checkpoint_path.rsplit('-',1)[1])
      # else:
      #   sess.run(init_op)

      print("{0} session ready".format(datetime.now().isoformat()))
      # log.info("{0} session ready".format(datetime.now().isoformat()))
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      step = 0
      # acc1=args.acc
      # n = 0
      tf_feed = TFNode.DataFeed(ctx.mgr, args.mode == "train" or args.mode == "retrain")
      while not sv.should_stop() and not tf_feed.should_stop() and step < args.steps:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        # using feed_dict
        batch_xs, batch_ys = feed_dict(tf_feed.next_batch(batch_size))
        feed = {x: batch_xs, y_: batch_ys,keep:dropout,is_training:True}
        if len(batch_xs) > 0:
          if args.mode == "train" or args.mode == "retrain":
            # _, summary, step = sess.run([train_op, summary_op, global_step], feed_dict=feed)
            _, step = sess.run([train_op,  global_step], feed_dict=feed)
            '''
            if dropout > 0.2:
                if step%10000==0:dropout=dropout*0.85
            else:
                dropout=0.7
            '''
            """
            acc=sess.run(accuracy,{x: batch_xs, y_: batch_ys,keep:1.})
            if acc>acc1:
              if flag and acc>0.9:
                os.popen('hdfs dfs -rm -r '+logdir+'/*') # 清空hdfs上面文件夹下的所有文件
                flag=False
              # acc1=acc # 训练达到一定程度加上
              saver.save(sess,logdir+'/'+args.model_name,global_step=step)
              n=0
              # learning_rate=1e-3
              # dropout=.7
            else:
              n += 1
              if n > 100:
                ckpt1 = tf.train.get_checkpoint_state(logdir)
                if ckpt1 and ckpt1.model_checkpoint_path:
                  saver.restore(sess, ckpt1.model_checkpoint_path)
                if learning_rate > 1e-7:
                  # learning_rate = learning_rate * .96**(step/10)
                  learning_rate = learning_rate * .8
                else:
                  learning_rate = 1e-3
                if dropout > 0.2:
                  dropout = dropout * .85
                else:
                  dropout = .7
            """

            # print accuracy and save model checkpoint to HDFS every 100 steps
            if (step % 100 == 0):
              [y1,loss1]= sess.run([y,loss], {x: batch_xs,y_:batch_ys, keep: 1., is_training: True})
              prediction = [1. if abs(x3 - 1) < abs(x3 - 0) else 0. for x1 in y1 for x2 in x1 for x3 in x2]
              prediction = np.reshape(prediction, [-1, IMAGE_PIXELS,IMAGE_PIXELS]).astype(np.uint8)
              accuracy = np.mean(np.equal(prediction, batch_ys).astype(np.float32))

              print("{0} step: {1} accuracy: {2} loss:{3}".format(datetime.now().isoformat(), step, accuracy,loss1))
              # log.info("{0} step: {1} accuracy: {2}".format(datetime.now().isoformat(), step, sess.run(accuracy,{x: batch_xs, y_: batch_ys})))
            if sv.is_chief:
              pass
              # summary_writer.add_summary(summary, step)
          else: # args.mode == "inference"
            """
            labels, preds, acc = sess.run([label, prediction, accuracy], feed_dict=feed)

            results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l,p in zip(labels,preds)]
            tf_feed.batch_results(results)
            print("acc: {0}".format(acc))
            """
            # log.info("acc: {0}".format(acc))
      if sv.should_stop() or step >= args.steps:
        tf_feed.terminate()

    # Ask for all the services to stop.
    print("{0} stopping supervisor".format(datetime.now().isoformat()))
    # log.info("{0} stopping supervisor".format(datetime.now().isoformat()))
    sv.stop()

