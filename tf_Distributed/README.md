参考：https://tensorflow.google.cn/deploy/distributed


----------
# Hello distributed TensorFlow!

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
`tf.train.Server.create_local_server`方法使用进程内服务器创建单进程群集。


----------


# Create a cluster
TensorFlow“集群”是参与TensorFlow图的分布式执行的一组“任务”。 每个任务都与一个TensorFlow“服务器”相关联，该服务器包含一个可用于创建会话的“master”和一个在图中执行操作的“worker”。 一个集群也可以分成一个或多个“jobs”，每个工作包含一个或多个tasks。

要创建群集，请在群集中为<font color=#d000 size=5>每个任务启动一个TensorFlow服务器</font>。 每个任务通常运行在不同的机器上，但是可以在同一台机器上运行多个任务（例如，控制不同的GPU设备）。 在每个任务中，执行以下操作：

 - 创建一个描述集群中所有任务的`tf.train.ClusterSpec`。 这应该每个任务是相同的。
 - 创建一个`tf.train.Server`，将`tf.train.ClusterSpec`传递给构造函数，并使用job 名称和task 索引标识本地任务。

## 创建一个tf.train.ClusterSpec来描述集群
群集规范字典将作业名称映射到网络地址列表。 将此字典传递给`tf.train.ClusterSpec`构造函数。 例如：

![这里写图片描述](http://img.blog.csdn.net/20171128095223253?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 在每个任务中创建一个tf.train.Server实例
`tf.train.Server`对象包含一组本地设备，一组到`tf.train.ClusterSpec`中的其他任务的连接，以及一个可用于执行分布式计算的tf.Session。 每个服务器都是特定命名作业的成员，并且在该作业中有一个任务索引。 服务器可以与群集中的任何其他服务器进行通信。

例如，要启动运行在localhost：2222和localhost：2223上的两台服务器的集群，请在本地计算机上的两个不同进程中运行以下代码片段：

```python
# In task 0:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)
```

```python
# In task 1:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)
```

注意：手动指定这些集群规范可能很乏味，特别是对于大型集群。 我们正在开发以编程方式启动任务的工具，例如 使用像Kubernetes这样的集群管理器。 如果您希望看到支持的特定集群管理器，请提出GitHub问题。


----------
# Specifying distributed devices in your model
要将操作放在特定进程上，可以使用相同的`tf.device`函数来指定ops是否在CPU或GPU上运行。 例如：

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

在上面的示例中，变量是在ps作业中的两个任务上创建的，模型的计算密集型部分是在`worker`作业中创建的。 TensorFlow将在jobs 之间插入适当的数据传输（从ps到worker，正向传递，worker从ps到应用渐变）。


----------
# Replicated training
被称为“数据并行性”的通用训练配置涉及`worker`任务中的多个任务，在不同的小批量数据上训练相同的模型，更新托管在ps作业中的一个或多个任务中的共享参数。 所有任务通常在不同的机器上运行。 在TensorFlow中有很多方法可以指定这个结构，我们正在构建一个库来简化指定复制模型的工作。 可能的方法包括：

 - **In-graph replication.** 在这种方法中，客户端构建一个包含一组参数的tf.Graph（在tf.Variable节点中固定为/ job：ps）; 以及模型的计算密集型部分的多个副本，每个副本都固定在/ job：worker中的不同任务。
 - **Between-graph replication.** 在这种方法中，每个`/job:worker`都有一个单独的客户端：工作者任务，通常与工作任务处于同一个进程中。 每个客户端都使用`tf.train.replica_device_setter`确定性地将它们映射到相同的任务中，然后构建一个类似的包含参数的图形（固定到/ job：ps。 以及模型的计算密集型部分的单个副本，固定到/ job：worker中的本地任务。
 - **Asynchronous training.** 在这种方法中，图的每个副本都有一个独立的训练循环，无需协调即可执行。 它与以上两种复制形式兼容。
 - **Synchronous training.** 在这种方法中，所有副本都读取当前参数的相同值，并行计算梯度，然后将它们应用到一起。 它与 **in-graph replication**（例如使用CIFAR-10多GPU训练器中的梯度平均）以及**Between-graph replication**（例如使用tf.train.SyncReplicasOptimizer）兼容。

## Putting it all together: example trainer program
以下代码显示了分布式训练程序的框架，实现了图间复制（between-graph replication）和异步培训（asynchronous training）。 它包含参数服务器和辅助任务的代码。

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

要使用两个参数服务器和两个workers启动训练，请使用以下命令行（假设脚本名为trainer.py）：

```python
# On ps0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=0
# On ps1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=1
# On worker0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=0
# On worker1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=1
```


----------
# Glossary
**Client**
客户端通常是一个构建TensorFlow图并构造tensorflow :: Session以与集群进行交互的程序。 客户端通常使用Python或C ++编写。 一个客户端进程可以直接与多台TensorFlow服务器交互（参见上面的“Replicated training”），一台服务器可以为多个客户端提供服务。

**Cluster**
TensorFlow集群包含一个或多个“jobs”，每个“job”分为一个或多个“tasks”列表。 一个集群通常专门用于特定的高级目标，比如训练一个神经网络，并行使用多台机器。 一个集群由一个`tf.train.ClusterSpec`对象定义。

**Job**
一个job 包括一份“tasks”清单，通常用于一个共同的目的。 例如，名为ps的作业（对于“参数服务器”）通常承载存储和更新变量的节点; 而名为`Worker`的作业通常承载执行计算密集型任务的无状态节点。 作业中的任务通常运行在不同的机器上。 job 角色是灵活的：例如，worker 可以保持某种状态。

**Master service**
RPC服务，提供对一组分布式设备的远程访问，并充当会话目标。 主服务实现`tensorflow :: Session`接口，负责跨一个或多个“worker services”协调工作。 所有的TensorFlow服务器都实现主服务。

**Task**
task 对应于特定的TensorFlow服务器，并且通常对应于单个进程。 任务属于某个特定的“job”，并由该工作任务列表中的索引来标识。

TensorFlow服务器运行作为集群成员的tf.train.Server实例的进程，并导出“master service”和“worker service”。

**Worker service**
使用本地设备执行TensorFlow图形部分的RPC服务。 一个worker 服务实现worker_service.proto。 所有的TensorFlow servers 都实现了worker服务。

