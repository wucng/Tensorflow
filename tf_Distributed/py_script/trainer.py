import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

        # Build model ...
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

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

        # global_step = tf.Variable(0, name='global_step', trainable=False)
        global_step = tf.contrib.framework.get_or_create_global_step()

        train_op = tf.train.AdagradOptimizer(0.5).minimize(
            cross_entropy, global_step=global_step)

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()

        init_op = tf.global_variables_initializer()


    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        # mon_sess.run(train_op)
        step = 0
        if step < 2000:
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, step = mon_sess.run([train_op, global_step], feed_dict={x: batch_xs, y_: batch_ys})
            # print("Step %d in task %d" % (step, FLAGS.task_index))
            if step % 100 == 0:
                print("accuracy: %f" % mon_sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                     y_: mnist.test.labels}))
        break

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

  # python3 trainer.py --ps_hosts=192.168.146.137:2226 --worker_hosts=192.168.146.133:2227,192.168.146.136:2228 --job_name="worker" --task_index=0
