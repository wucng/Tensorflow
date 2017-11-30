#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
结合tf spark （集群版）（也可以读取本地路径的数据）
1、先将数据转成hdfs（pickle） 参考：mnist 转成 hdfs
2、从hdfs上读取pickle数据转成numpy 参考：hdfs 转 numpy
3、将得到的数据（numpy）在集群上使用tf训练

tf 集群运行 参考：https://github.com/fengzhongyouxia/Tensorflow/blob/master/tf_Distributed/py_script/mnist_dist.py

mnist 转成 hdfs  参考：https://github.com/fengzhongyouxia/Tensorflow/blob/master/tf_spark/README.md
hdfs 转 numpy 参考：http://blog.csdn.net/wc781708249/article/details/78251701#t3
tensorflow 读取hdfs https://github.com/fengzhongyouxia/Tensorflow/blob/master/tf_Distributed/Read_HDFS.md

集群命令：
spark-submit mnist_dist.py --ps_hosts=10.0.100.25:2220 --worker_hosts=10.0.100.14:2221,10.0.100.15:2222 --job_name="ps" --task_index=0
spark-submit mnist_dist.py --ps_hosts=10.0.100.25:2220 --worker_hosts=10.0.100.14:2221,10.0.100.15:2222 --job_name="worker" --task_index=0
spark-submit mnist_dist.py --ps_hosts=10.0.100.25:2220 --worker_hosts=10.0.100.14:2221,10.0.100.15:2222 --job_name="worker" --task_index=1


单机执行命令：spark-submit test.py

或

spark-submit \
--queue default \
--num-executors 45 \
--executor-memory 2G \
--driver-memory 12G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
test.py

或

spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 45 \
--executor-memory 2G \
--driver-memory 12G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
test.py

"""


from pyspark import SparkContext,SparkConf
import numpy as np
import tensorflow as tf
# import pickle

index=["part-00000","part-00001","part-00002","part-00003","part-00004","part-00005","part-00006",
       "part-00007","part-00008","part-00009"]
mode=["train","test"]
img_label=["images","labels"]

dirPath='hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/mnist/pickle/' # 注该数据为pickle格式

# 解析路径
train_img_paths=[]
[train_img_paths.append(dirPath+mode[0]+'/'+img_label[0]+'/'+i) for i in index]
train_label_paths=[]
[train_label_paths.append(dirPath+mode[0]+'/'+img_label[1]+'/'+i) for i in index]

test_img_paths=[]
[test_img_paths.append(dirPath+mode[1]+'/'+img_label[0]+'/'+i) for i in index]
test_label_paths=[]
[test_label_paths.append(dirPath+mode[1]+'/'+img_label[1]+'/'+i) for i in index]

sc = SparkContext(conf=SparkConf().setAppName("The first example"))
# textFiles=sc.textFile(dirPath) # 读取 txt，csv 格式数据
def get_data(paths):
    for i,train_img_path in enumerate(paths):
        textFiles=sc.pickleFile(train_img_path) # 读取pickle数据
        data=textFiles.collect()
        if i==0:
            data1 = np.array(data, np.float32)  # 转成array
        else:
            data1=np.vstack((data1,np.array(data, np.float32)))
    return data1

train_x=get_data(train_img_paths) # (60000, 784)
train_y=get_data(train_label_paths) # (60000, 10)

test_x=get_data(test_img_paths) # (10000, 784)
test_y=get_data(test_label_paths) # (10000, 10)

train_data=np.hstack((train_x,train_y))
np.random.shuffle(train_data)

# test_data=np.hstack((test_x,test_y))
# np.random.shuffle(test_data)

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
    img = data1[:, 0:-10]
    img = img.astype(float)

    label = data1[:, -10:]
    label = label.astype(float)  # 类型转换

    return img,label


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

            # Create the model
            x = tf.placeholder(tf.float32, [None, 784])
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            y = tf.matmul(x, W) + b
            # y=tf.nn.softmax(tf.nn.relu(y))
            y = tf.nn.softmax(y)
            # Define loss and optimizer
            y_ = tf.placeholder(tf.float32, [None, 10])
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = tf.train.AdagradOptimizer(0.5).minimize(
                cross_entropy, global_step=global_step)

            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()

            init_op = tf.global_variables_initializer()

        # Create a "Supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 # logdir="./opt", # 本地路径
                                 logdir="hdfs://xxx:8020/user/root/mnist_model",  # 换成hdfs路径
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization and restoring from
        # a checkpoint.
        sess = sv.prepare_or_wait_for_session(server.target)

        # Start queue runners for the input pipelines (if ang).
        sv.start_queue_runners(sess)

        # Loop until the supervisor shuts down (or 2000 steps have completed).
        step = 0
        while not sv.should_stop() and step < 2000:
            # batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_xs, batch_ys = next_batch(train_data, 100)
            _, step = sess.run([train_op, global_step], feed_dict={x: batch_xs, y_: batch_ys})
            # print("Step %d in task %d" % (step, FLAGS.task_index))
            if step % 100 == 0:
                print("accuracy: %f" % sess.run(accuracy, feed_dict={x: test_x,
                                                                     y_: test_y}))

if __name__ == "__main__":
    tf.app.run()
