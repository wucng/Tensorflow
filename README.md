参考：
1、https://github.com/tensorflow/serving
2、https://tensorflow.google.cn/
3、https://www.tensorflow.org/


----------
tf_serving 安装参考：http://blog.csdn.net/wc781708249/article/details/78594750
tf_serving 模型部署：http://blog.csdn.net/wc781708249/article/details/78596459


----------
本篇文章是对官网提供的样例，[mnist_saved_model](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example)进行全方位的解析，最终达到使用tf_serving 可以自由搭建自己想要的模型。

主要解析 [example](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example)里的mnist_input_data.py，mnist_saved_model.py，mnist_client.py，BUILD


----------
# 测试
先来测试下是否mnist_saved_model只依赖于
mnist_input_data.py，mnist_saved_model.py，mnist_client.py，BUILD
```
cd ~/serving/tensorflow_serving
mkdir test # 最好在tensorflow_serving新建目录，放在其他目录下会报错
cp -r example/mnist* example/BUILD test
# cd test
```
参考：[TF_Serving 部署一个模型](http://blog.csdn.net/wc781708249/article/details/78596459) 部署模型

主要命令：

```python
# 补充 /tmp 为临时目录，里面的文件会自动清空的，如果想永久保存模型，需换一个其他目录，这里只是为了练习需要，后续将会换一个目录，便于模型保存管理和发布
# 模型保存目录和数据保存目录可以是其他目录

rm -rf /tmp/mnist_model

# Train And Export TensorFlow Model
cd ~/serving
# 这里要将example目录 都换成 test
#$>bazel build -c opt //tensorflow_serving/example:mnist_saved_model
$>bazel build -c opt //tensorflow_serving/test:mnist_saved_model

$>bazel-bin/tensorflow_serving/test/mnist_saved_model /tmp/mnist_model
# 或者 如果安装了tensorflow-serving-api，则可以运行
python tensorflow_serving/test/mnist_saved_model.py /tmp/mnist_model


----------


# Load Exported Model With Standard TensorFlow ModelServer
cd ~/serving
$>bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server

$>bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/
# 或者
$>tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/


----------


# Test The Server
cd ~/serving
# $>bazel build -c opt //tensorflow_serving/example:mnist_client
$>bazel build -c opt //tensorflow_serving/test:mnist_client

$>bazel-bin/tensorflow_serving/test/mnist_client --num_tests=1000 --server=localhost:9000
# 或者
$>python tensorflow_serving/test/mnist_client.py --num_tests=1000 --server=localhost:9000
```

![这里写图片描述](http://img.blog.csdn.net/20171124135129613?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20171124140212535?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20171124140521526?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20171124140655594?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

----------
# 修改mnist_saved_model

## mnist_saved_model.py

```python
# top_k的使用
import tensorflow as tf
import numpy as np

y=[[0,0,1,0],[1,0,0,0],[0,1,0,0]] # 对应的标签为[2,0,1]
values, indices = tf.nn.top_k(y, 4)
table = tf.contrib.lookup.index_to_string_table_from_tensor(
  tf.constant([str(i) for i in range(4)]))
prediction_classes = table.lookup(tf.to_int64(indices))

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
sess.run(tf.tables_initializer())
pred=sess.run(prediction_classes)

print(pred)
print(type(pred))
print(list(map(int,pred[:,0])))

"""
[[b'2' b'0' b'1' b'3']
 [b'0' b'1' b'2' b'3']
 [b'1' b'0' b'2' b'3']]
<class 'numpy.ndarray'>
[2, 0, 1]
"""

```

```python
cd ~/serving/tensorflow_serving
mkdir test # 最好在tensorflow_serving新建目录，放在其他目录下会报错
cp -r example/mnist* example/BUILD test

```

修改过的[mnist_saved_model.py](https://github.com/fengzhongyouxia/Tensorflow/blob/master/mnist_saved_model_2.py)

```python
# Train And Export TensorFlow Model
cd /home/wu/Downloads/serving
# bazel build -c opt ///home/wu/pytest/test:mnist_saved_model
bazel build -c opt //tensorflow_serving/test:mnist_saved_model

# Train TensorFlow Model
python tensorflow_serving/test/mnist_saved_model.py
```

# mnist_client.py

```python
# Load Exported Model With Standard TensorFlow ModelServer
cd ~/serving
$>bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server

$>bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/home/wu/pytest/MNIST/mnist_model/

----------

# Test The Server
cd ~/serving
$>bazel build -c opt tensorflow_serving/test:mnist_client
$>python tensorflow_serving/test/mnist_client.py --num_tests=1000 --server=localhost:9000
```
