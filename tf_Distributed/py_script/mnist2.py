# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# --------------------按模型修改----------------------------
# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 20,
                            'Steps to validate and print loss')
# --------------------------------------------------------

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # -----------------------按模型修改----------------------------------------
            # Build model ...
            mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

            # Create the model
            x = tf.placeholder(tf.float32, [None, 784])
            W = tf.Variable(tf.random_normal([784, 10]))
            b = tf.Variable(tf.random_normal([10]))
            y = tf.nn.softmax(tf.nn.relu(tf.matmul(x, W) + b))

            y=tf.nn.softmax(y)
            # Define loss and optimizer
            y_ = tf.placeholder(tf.float32, [None, 10])
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = tf.train.AdagradOptimizer(0.5).minimize(
                cross_entropy, global_step=global_step)

            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            grads_and_vars = optimizer.compute_gradients(cross_entropy)

            # -----------------------------------------------------------------#
            if issync == 1:
                # 同步模式计算更新梯度
                rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=len(
                                                            worker_hosts),
                                                        replica_id=FLAGS.task_index,
                                                        total_num_replicas=len(
                                                            worker_hosts),
                                                        use_locking=True)
                train_op = rep_op.apply_gradients(grads_and_vars,
                                                  global_step=global_step)
                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()
            else:
                # 异步模式计算更新梯度
                train_op = optimizer.apply_gradients(grads_and_vars,
                                                     global_step=global_step)

            init_op = tf.global_variables_initializer()

            saver = tf.train.Saver()
            tf.summary.scalar('cost', cross_entropy)
            summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="./checkpoint/", # 模型保存位置
                                 init_op=init_op, # 变量初始化
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60) # 每5s保存一次

        with sv.prepare_or_wait_for_session(server.target) as sess:
            # 如果是同步模式
            if FLAGS.task_index == 0 and issync == 1:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
            # -------------------按模型修改--------------------------

            step = 0
            while not sv.should_stop() and step < 2000:
                batch_xs, batch_ys = mnist.train.next_batch(100)
                _, step = sess.run([train_op, global_step], feed_dict={x: batch_xs, y_: batch_ys})
                # print("Step %d in task %d" % (step, FLAGS.task_index))
                if step % steps_to_validate == 0:
                    print("step:",step,"accuracy: %f" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                     y_: mnist.test.labels}))

            # step = 0
            # while step < 10000:
            #     train_x = np.random.randn(1)
            #     train_y = 2 * train_x + np.random.randn(1) * 0.33 + 10
            #     _, loss_v, step = sess.run([train_op, loss_value, global_step],
            #                                feed_dict={input: train_x, label: train_y})
            #     if step % steps_to_validate == 0:
            #         w, b = sess.run([weight, biase])
            #         print("step: %d, weight: %f, biase: %f, loss: %f" % (step, w, b, loss_v))
            # --------------------------------------------------------------
        sv.stop()


# def loss(label, pred):
#     return tf.square(label - pred)


if __name__ == "__main__":
    tf.app.run()

# python3 mnist_dist.py --ps_hosts=192.168.146.137:2222 --worker_hosts=192.168.146.133:2224,192.168.146.136:2225 --job_name="worker" --task_index=0

# python3 mnist2.py --ps_hosts=192.168.146.133:2222 --worker_hosts=192.168.146.136:2224 --job_name="worker" --task_index=0
