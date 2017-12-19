# -*- coding: UTF-8 -*-

'''
--------------集群版----------------
train=1 训练
train=-1 推理
单机测试：
python mnist_dist.py --ps_hosts=10.0.100.14:2220 --worker_hosts=10.0.100.14:2221 --job_name="ps" --task_index=0
python mnist_dist.py --ps_hosts=10.0.100.14:2220 --worker_hosts=10.0.100.14:2221 --job_name="worker" --task_index=0

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
xxx.py --ps_hosts= ……
集群命令：
python mnist_dist.py --ps_hosts=10.0.100.25:2220 --worker_hosts=10.0.100.14:2221,10.0.100.15:2222 --job_name="ps" --task_index=0
python mnist_dist.py --ps_hosts=10.0.100.25:2220 --worker_hosts=10.0.100.14:2221,10.0.100.15:2222 --job_name="worker" --task_index=0
python mnist_dist.py --ps_hosts=10.0.100.25:2220 --worker_hosts=10.0.100.14:2221,10.0.100.15:2222 --job_name="worker" --task_index=1
或
spark-submit mnist_dist.py --ps_hosts=10.0.100.25:2220 --worker_hosts=10.0.100.14:2221,10.0.100.15:2222 --job_name="ps" --task_index=0
spark-submit mnist_dist.py --ps_hosts=10.0.100.25:2220 --worker_hosts=10.0.100.14:2221,10.0.100.15:2222 --job_name="worker" --task_index=0
spark-submit mnist_dist.py --ps_hosts=10.0.100.25:2220 --worker_hosts=10.0.100.14:2221,10.0.100.15:2222 --job_name="worker" --task_index=1
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
try:
  from osgeo import gdal,ogr
except:
  import gdal,ogr

import os
from os import path
import datetime
import walter_model

# from numba.decorators import jit
# import multiprocessing
# from PIL import Image
from osgeo.gdalconst import *
import glob

start_time = datetime.datetime.now()
print("startTime: ", start_time)


tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

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

            m = walter_model.Model(walter_model.train).model()

            summary_op = tf.summary.merge_all()

            # 初始化所有的op
            init = tf.global_variables_initializer()

            saver = tf.train.Saver()

        logdir = "hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/model_walter"
        # Create a "Supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=logdir,
                                 init_op=init,
                                 # summary_op=None,
                                 saver=saver,
                                 # saver=None, # None 不自动保存模型
                                 # recovery_wait_secs=1,
                                 global_step=walter_model.global_step,
                                 stop_grace_secs=300,
                                 save_model_secs=10,
                                 checkpoint_basename='save_net.ckpt')

        # The supervisor takes care of session initialization and restoring from
        # a checkpoint.
        sess = sv.prepare_or_wait_for_session(server.target)

        # Start queue runners for the input pipelines (if ang).
        sv.start_queue_runners(sess)
        if walter_model.train == 1:  # train
            while not sv.should_stop():
                walter_model.Train_and_inference_model(sess,m).Train()
                break
            sess.close()
            exit()

        if walter_model.train==-1: # inference
            while not sv.should_stop():
                walter_model.Train_and_inference_model(sess,m).Inference(FLAGS.task_index)
                break
            sess.close()
            exit()

if __name__ == "__main__":
    tf.app.run()
