# Hello distributed TensorFlow!

To see a simple TensorFlow cluster in action, execute the following:

```python
# Start a TensorFlow server as a single-process "cluster".
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.train.Server.create_local_server()
>>> sess = tf.Session(server.target)  # Create a session on the server.
>>> sess.run(c)
'Hello, distributed TensorFlow!'
```

The `tf.train.Server.create_local_server` method creates a single-process cluster, with an in-process server.

--------------------------------------------------------------------------------

```python
# -*- coding: UTF-8 -*-
import tensorflow as tf

c = tf.constant("Hello, distributed TensorFlow!")
server = tf.train.Server.create_local_server() # 启动本机的本地服务器
sess = tf.Session(server.target)  # Create a session on the server.

with tf.device("/job:local/task:0"),tf.device("/cpu:0"): #使用本地第一个工作节点，第一个cpu
    print(sess.run(c)) # ===> b'Hello, distributed TensorFlow!'
sess.close()
```

--------------------------------------------------------------------------------

```python
# -*- coding: UTF-8 -*-
import tensorflow as tf

c = tf.constant("Hello, distributed TensorFlow!")
b=tf.get_variable("c",[2,2],tf.float32,initializer=tf.random_normal_initializer)

server = tf.train.Server.create_local_server() # 启动本机的本地服务器
sess = tf.Session(server.target)  # Create a session on the server.

sess.run(tf.global_variables_initializer())

with tf.device("/job:local/task:0"),tf.device("/cpu:0"): #使用本地第1个工作节点，第1个cpu
    print(sess.run(c)) # ===> b'Hello, distributed TensorFlow!'

with tf.device("/job:local/task:1"),tf.device("/cpu:1"): #使用本地第2个工作节点，第2个cpu
    print(sess.run(b)) # ===>

with tf.device("/job:local/task:2"),tf.device("/cpu:2"): #使用本地第3个工作节点，第3个cpu
    print(sess.run(c)) # ===> b'Hello, distributed TensorFlow!'

with tf.device("/job:local/task:3"),tf.device("/cpu:3"): #使用本地第4个工作节点，第4个cpu
    print(sess.run(c)) # ===> b'Hello, distributed TensorFlow!'

with tf.device("/job:local/task:4"),tf.device("/cpu:4"): #使用本地第5个工作节点，第5个cpu
    print(sess.run(c)) # ===> b'Hello, distributed TensorFlow!'
sess.close()
```

--------------------------------------------------------------------------------

```python
# -*- coding: UTF-8 -*-
import tensorflow as tf

with tf.device("/job:local/task:0"),tf.device("/cpu:0"): #使用本地第1个工作节点，第1个cpu
    c = tf.constant("Hello, distributed TensorFlow!")
with tf.device("/job:local/task:0"),tf.device("/cpu:0"): #使用本地第1个工作节点，第1个cpu
    b=tf.get_variable("b",[2,2],tf.float32,initializer=tf.random_normal_initializer)

server = tf.train.Server.create_local_server() # 启动本机的本地服务器
sess = tf.Session(server.target)  # Create a session on the server.

sess.run(tf.global_variables_initializer())

print(sess.run(c))
print(sess.run(b))

sess.close()
```

--------------------------------------------------------------------------------

```python
# -*- coding: utf-8 -*-

import tensorflow as tf

server = tf.train.Server.create_local_server() # 启动本机的本地服务器
sess = tf.Session(server.target)  # Create a session on the server.

with tf.device("/job:local/task:0"),tf.device("/cpu:0"): #使用本地第1个工作节点，第1个cpu
    c = tf.constant("Hello, distributed TensorFlow!")
with tf.device("/job:local/task:0"),tf.device("/gpu:0"): #使用本地第1个工作节点，第1个gpu
    b=tf.get_variable("b",[2,2],tf.float32,initializer=tf.random_normal_initializer)

sess.run(tf.global_variables_initializer())

print(sess.run(c))
print(sess.run(b))

sess.close()
```

--------------------------------------------------------------------------------

```python
# -*- coding: utf-8 -*-

import tensorflow as tf

server = tf.train.Server.create_local_server() # 启动本机的本地服务器
sess = tf.Session(server.target)  # Create a session on the server.

with tf.device("/cpu:0"): #使用本地第1个工作节点，第1个cpu
    c = tf.constant("Hello, distributed TensorFlow!")
with tf.device("/gpu:0"): #使用本地第1个工作节点，第1个cpu
    b=tf.get_variable("b",[2,2],tf.float32,initializer=tf.random_normal_initializer)

sess.run(tf.global_variables_initializer())

print(sess.run(c))
print(sess.run(b))

sess.close()
```


----------


# Create a cluster
A TensorFlow "cluster" is a set of "tasks" that participate in the distributed execution of a TensorFlow graph. Each task is associated with a TensorFlow "server", which contains a "master" that can be used to create sessions, and a "worker" that executes operations in the graph. A cluster can also be divided into one or more "jobs", where each job contains one or more tasks.

<center>![这里写图片描述](http://img.blog.csdn.net/20171128102717538?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

To create a cluster, you start one TensorFlow server per task in the cluster. Each task typically runs on a different machine, but you can run multiple tasks on the same machine (e.g. to control different GPU devices). In each task, do the following:

  1. Create a tf.train.ClusterSpec that describes all of the tasks in the cluster. This should be the same for each task.
  2. Create a tf.train.Server, passing the tf.train.ClusterSpec to the constructor, and identifying the local task with a job name and task index.

Create a tf.train.ClusterSpec to describe the cluster

![](http://img.blog.csdn.net/20171128102752227?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

注： 端口号，必须是2222以上，比如2222，2223,2224,2225等 而不是机器本身的端口号

# Create a tf.train.Server instance in each task
A tf.train.Server object contains a set of local devices, a set of connections to other tasks in itstf.train.ClusterSpec, and a tf.Session that can use these to perform a distributed computation. Each server is a member of a specific named job and has a task index within that job. A server can communicate with any other server in the cluster.

For example, to launch a cluster with two servers running on `localhost:2222` and `localhost:2223`, run the following snippets in two different processes on the local machine:

```python
# In task 0:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)

# In task 1:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)
```

**Note**: Manually specifying these cluster specifications can be tedious, especially for large clusters. We are working on tools for launching tasks programmatically, e.g. using a cluster manager like Kubernetes. If there are particular cluster managers for which you'd like to see support, please raise a GitHub issue.

# Specifying distributed devices in your model
To place operations on a particular process, you can use the same tf.device function that is used to specify whether ops run on the CPU or GPU. For example:

```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

with tf.Session("grpc://worker7.example.com:2222") as sess:
  for _ in range(10000):
    sess.run(train_op)
```

<font color=#d000 size=5>**注**：</font> 通常job：ps对应cpu，只负责创建变量，汇总梯度；job：worker对应gpu，负责数据传输、图表构建与数据计算

In the above example, the variables are created on two tasks in the ps job, and the compute-intensive part of the model is created in the worker job. TensorFlow will insert the appropriate data transfers between the jobs (from psto worker for the forward pass, and from worker to ps for applying gradients).

In-graph replication.
Between-graph replication.
Asynchronous training
Synchronous training.

Putting it all together: example trainer program

The following code shows the skeleton of a distributed trainer program, implementing between-graph replicationand asynchronous training. It includes the code for the parameter server and worker tasks.

```python
import argparse
import sys

import tensorflow as tf

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

      # Build model...
      loss = ...
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

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
        mon_sess.run(train_op)

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
```
