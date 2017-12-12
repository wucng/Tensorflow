参考：
1、https://tensorflow.google.cn/
2、https://www.tensorflow.org/
3、
http://note.youdao.com/noteshare?id=92af08c5ceff28db9748f822fee05322&sub=7E5B724B2443405C9DD3D8E22C3A1EE9

----------
TF_Serving安装参考：http://blog.csdn.net/wc781708249/article/details/78594750


----------
本教程将向您介绍如何使用TensorFlow Serving组件导出经过训练的TensorFlow模型，并使用标准tensorflow_model_server来提供它。 如果您已经熟悉TensorFlow Serving，并且您想了解更多有关服务器内部工作方式的信息，请参阅[TensorFlow Serving高级教程](https://tensorflow.google.cn/serving/serving_advanced)。

本教程使用TensorFlow教程中介绍的简单Softmax回归模型进行手写图像（MNIST数据）分类。 如果您不知道TensorFlow或MNIST是什么，请参阅[MNIST For ML Beginners教程](http://tensorflow.google.cn/tutorials/mnist/beginners/index.html#mnist-for-ml-beginners)。

本教程的代码由两部分组成：

 - 训练并导出模型的Python文件[mnist_saved_model.py](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py)。
 - ModelServer二进制文件，可以使用apt-get来安装，也可以从C ++文件（main.cc）编译。 TensorFlow
   Serving ModelServer发现新的导出模型并运行gRPC服务来提供服务。

开始之前，请完成[先决条件](https://tensorflow.google.cn/serving/setup#prerequisites)。

注意：下面的所有bazel构建命令都使用标准-c opt 标志。 要进一步优化构建，请参阅[此处](https://tensorflow.google.cn/serving/setup#optimized_build)的说明。


----------
docker部署：参考 http://geek.csdn.net/news/detail/194233
部署自己的模型：参考 https://zhuanlan.zhihu.com/p/23361413

----------
# Train And Export TensorFlow Model

正如你在mnist_saved_model.py中看到的那样，训练与MNIST For ML初学者教程中的相同。 TensorFlow graph 在TensorFlow session   `sess`中启动，输入张量（image）为x，输出张量（Softmax score）为y。

然后我们使用TensorFlow的[SavedModelBuilder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder.py)模块导出模型。 `SavedModelBuilder`将训练过的模型的“快照”保存到可靠的存储中，以便以后可以加载进行推理。

有关SavedModel格式的详细信息，请参阅[SavedModel README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)上的文档。

来自[mnist_saved_model.py](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py)，下面是一个简短的代码片段来说明将模型保存到磁盘的一般过程。

```
export_path_base = sys.argv[-1]
export_path = os.path.join(
      compat.as_bytes(export_path_base),
      compat.as_bytes(str(FLAGS.model_version)))
print 'Exporting trained model to', export_path
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           'predict_images':
               prediction_signature,
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               classification_signature,
      },
      legacy_init_op=legacy_init_op)
builder.save()
```
`SavedModelBuilder.__init__` 接收以下参数:

 - `export_path`是导出目录的路径。

`SavedModelBuilder`将创建该目录，如果它不存在。 在这个例子中，我们连接命令行参数和`FLAGS.model_version`以获得导出目录。 `FLAGS.model_version`指定模型的版本。 导出相同模型的较新版本时，应该指定较大的整数值。 每个版本将被导出到给定路径下的不同子目录。

您可以使用以下参数使用`SavedModelBuilder.add_meta_graph_and_variables（）`将元图和变量添加到构建器：

 - `sess`是TensorFlow session ，持有你正在导出的训练模型。
 - tags 是用于保存元图形的一组标签。 在这种情况下，由于我们打算在服务中使用图形，因此我们使用来自预定义的SavedModel标签常量的服务标签。有关更多详细信息，请参阅[tag_constants.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/tag_constants.py)和相关的[TensorFlow API文档](https://tensorflow.google.cn/api_docs/python/tf/saved_model/tag_constants)。
 - `signature_def_map`指定将用户提供的密钥映射到tensorflow :: SignatureDef以添加到元图中。签名指定正在输出哪种类型的模型，以及运行推理时绑定的输入/输出张量。

特殊签名密钥`serving_default`指定默认的服务签名。 默认的服务签名def key以及与签名相关的其他常量被定义为SavedModel签名常量的一部分。 有关更多详细信息，请参阅[signature_constants.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)和相关的[TensorFlow 1.0 API文档](https://tensorflow.google.cn/api_docs/python/tf/saved_model/signature_constants)。

此外，为了帮助构建签名defs，SavedModel API提供了[签名def utils](https://tensorflow.google.cn/api_docs/python/tf/saved_model/signature_def_utils)。 具体来说，在上面的mnist_saved_model.py代码片段中，我们使用signature_def_utils.build_signature_def（）来构建predict_signature和classification_signature。

作为如何定义predict_signature的示例，util使用以下参数：

 - inputs = {'images'：tensor_info_x}指定输入张量信息。
 - outputs = {'scores'：tensor_info_y}指定scores张量信息。
 - method_name是用于推断的方法。 对于预测请求，应该设置为张量流/服务/预测。  对于其他方法名称，请参阅[signature_constants.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)和相关的TensorFlow 1.0 API文档。

请注意，tensor_info_x和tensor_info_y具有此处定义的tensorflow :: TensorInfo协议缓冲区的结构。 为了轻松构建张量信息，TensorFlow SavedModel API还提供了[utils.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/utils.py)和相关的TensorFlow 1.0 API文档。

另外请注意，images 和scores 是张量别名。 它们可以是任何你想要的唯一字符串，并且它们将成为后来发送预测请求时引用张量绑定的张量x和y的逻辑名称。

例如，如果x引用名称为“long_tensor_name_foo”的张量，并且y引用名称为“generated_tensor_name_bar”的张量，则构建器将张量逻辑名称存储为实名映射（'images' - >'long_tensor_name_foo'）和 ' - >'generated_tensor_name_bar'）。 这允许用户在运行推理时用它们的逻辑名称来引用这些张量。

注意：除了上面的描述之外，还可以在[这里](https://tensorflow.google.cn/serving/signature_defs)找到与签名def结构相关的文档以及如何设置它们。


----------
让我们来运行它！

清除导出目录，如果它已经存在：

```
$>rm -rf /tmp/mnist_model
```
如果您想安装tensorflow和tensorflow-serving-api PIP软件包，可以使用简单的python命令运行所有的Python代码（导出和客户端）。 要安装PIP包装，请按照[此处](https://tensorflow.google.cn/serving/setup#tensorflow_serving_python_api_pip_package)的说明进行操作。 也可以使用Bazel来构建必要的依赖关系，并运行所有代码而不安装这些包。 其余的codelab将有Bazel和PIP选项的说明。

Bazel:

```python
cd ~/serving
$>bazel build -c opt //tensorflow_serving/example:mnist_saved_model

$>bazel-bin/tensorflow_serving/example/mnist_saved_model /tmp/mnist_model
# 或者 如果您安装了tensorflow-serving-api，则可以运行
python tensorflow_serving/example/mnist_saved_model.py /tmp/mnist_model


Training model...

...

Done training!
Exporting trained model to /tmp/mnist_model
Done exporting!
```

现在我们来看看导出目录。

```
$>ls /tmp/mnist_model
1
```
如上所述，将创建一个子目录用于导出模型的每个版本。 FLAGS.model_version具有默认值1，因此创建相应的子目录1。

```
$>ls /tmp/mnist_model/1
saved_model.pb variables
```
每个版本的子目录包含以下文件：

 - saved_model.pb是序列化的tensorflow :: SavedModel。
   它包括一个或多个模型的图形定义，以及模型的元数据（如签名）。
 - variables 是包含图形的序列化变量的文件。

有了这个，你的TensorFlow模型被导出并准备好加载！


----------
# Load Exported Model With Standard TensorFlow ModelServer
如果您想使用本地编译的ModelServer，请运行以下命令：

```
cd ~/serving
$>bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server

$>bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/ &> mnist_log &
# 或者 参考下面的说明
$>tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/ &> mnist_log &
```
<font color=#d000 size=5>或者</font>，如果您希望跳过编译并[Installing using apt-get（在tf_serving安装时安装了这步骤）](https://tensorflow.google.cn/serving/setup#installing_using_apt-get)， 然后使用以下命令运行服务器：

```
tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/ &> mnist_log &
```


----------
# Test The Server

----------
我们可以使用提供的[mnist_client](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example/mnist_client.py)实用程序来测试服务器。 客户端下载MNIST测试数据，将其作为请求发送给服务器，并计算推理错误率。

运行Bazel：

```python
cd ~/serving
$>bazel build -c opt //tensorflow_serving/example:mnist_client

$>bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000
# 或者
$>python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=localhost:9000


...
Inference error rate: 10.5%
```
<font color=#d000 size=5>或者</font>，如果您安装了[这一步在安装TF_Serving(Installing using apt-get)](https://tensorflow.google.cn/serving/setup#installing_using_apt-get)，请运行：

```
python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=localhost:9000
```
我们预计训练的Softmax模型的准确率大约为91％，前1000张测试图像的推理错误率为10.5％。 这确认服务器加载并运行训练的模型成功！


----------


# 命令总结

```python
rm -rf /tmp/mnist_model

# Train And Export TensorFlow Model
cd ~/serving
$>bazel build -c opt //tensorflow_serving/example:mnist_saved_model

$>bazel-bin/tensorflow_serving/example/mnist_saved_model /tmp/mnist_model
# 或者 如果您安装了tensorflow-serving-api，则可以运行
python tensorflow_serving/example/mnist_saved_model.py /tmp/mnist_model

# Load Exported Model With Standard TensorFlow ModelServer
cd ~/serving
$>bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server

$>bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/
# 或者
$>tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/

# Test The Server
cd ~/serving
$>bazel build -c opt //tensorflow_serving/example:mnist_client

$>bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000
# 或者
$>python tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=localhost:9000
```
