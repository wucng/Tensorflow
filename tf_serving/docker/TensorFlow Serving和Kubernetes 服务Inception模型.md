参考：https://tensorflow.google.cn/serving/serving_inception


----------


本教程演示如何使用在Docker容器中运行的TensorFlow Serving组件来服务TensorFlow初始模型，以及如何使用Kubernetes部署服务集群。

要了解更多关于TensorFlow Serving的信息，我们推荐[TensorFlow Serving基本教程](https://tensorflow.google.cn/serving/serving_basic)和[TensorFlow Serving高级教程](https://tensorflow.google.cn/serving/serving_advanced)。

要了解有关TensorFlow初始模型的更多信息，我们推荐在[TensorFlow in Inception](https://github.com/tensorflow/models/tree/master/research/inception)。

 - 第0部分展示了如何创建一个TensorFlow Serving Docker镜像用于部署
 - 第1部分展示了如何在本地容器中运行图像。
 - 第2部分展示了如何在Kubernetes中部署。

# 第0部分：创建一个Docker镜像
有关构建TensorFlow Serving Docker镜像的详细信息，请参阅通过[Docker使用TensorFlow Serving](https://tensorflow.google.cn/serving/docker)。

## 运行容器
我们使用[Dockerfile.devel](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel)创建一个基于镜像的$ USER / tensorflow-serving-devel。 然后使用构建的图像在本地启动一个容器。

```python
$ docker build --pull -t $USER/tensorflow-serving-devel -f tensorflow_serving/tools/docker/Dockerfile.devel .
# 或
$ docker build --pull -t $USER/tensorflow-serving-devel -f Dockerfile.devel .
# 或
$ docker build -t $USER/tensorflow-serving-devel . # 保存成Dockerfile文件
$ docker run --name=inception_container -it $USER/tensorflow-serving-devel

# 或 将Dockerfile.devel中的内容保存成 Dockerfile
docker build -t tensorflow-serving:v1 .
git clone --recurse-submodules https://github.com/tensorflow/serving # serving下载到主机上
docker run -it -v /home/wu/serving:/serving tensorflow-serving:v1 /bin/bash # 使用-v 挂载到容器中

# 或
docker pull registry.cn-hangzhou.aliyuncs.com/781708249/tensorflow-serving:v1 # 已经配置好的tensorflow serving 从阿里镜像拉下来
git clone --recurse-submodules https://github.com/tensorflow/serving # serving下载到主机上
docker run -it -v /home/wu/serving:/serving registry.cn-hangzhou.aliyuncs.com/781708249/tensorflow-serving:v1 /bin/bash # 使用-v 挂载到容器中
```
## 在容器中克隆，配置和构建TensorFlow服务
注意：下面的所有bazel构建命令都使用标准-c选项标志。 要进一步优化构建，请参阅[此处](https://tensorflow.google.cn/serving/setup#optimized_build)的说明。

在正在运行的容器中，我们克隆，配置和构建TensorFlow服务示例代码。

```
root@c97d8e820ced:/# git clone --recurse-submodules https://github.com/tensorflow/serving  # 从主机中挂载了serving 可以跳过这步
root@c97d8e820ced:/# cd serving/tensorflow
root@c97d8e820ced:/serving/tensorflow# ./configure
root@c97d8e820ced:/serving# cd ..
root@c97d8e820ced:/serving# bazel build -c opt tensorflow_serving/example/...
```
接下来，我们可以使用[这里](https://tensorflow.google.cn/serving/setup#installing_using_apt-get)的指令安装一个TensorFlow ModelServer和apt-get，或者使用下面的代码构建一个ModelServer二进制文件：

```
root@c97d8e820ced:/serving# bazel build -c opt tensorflow_serving/model_servers:tensorflow_model_server
```
本教程的其余部分假定您在本地编译ModelServer，在这种情况下运行它的命令是`bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server`。 但是，如果使用apt-get安装ModelServer，只需用`tensorflow_model_server`替换该命令即可。

## 在容器中导出初始模型
在正在运行的容器中，我们运行[inception_saved_model.py](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_saved_model.py)使用发布的[Inception模型训练检查点](http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz)导出初始模型。 而不是从头开始训练，我们使用训练有素的变量的现成检查点来恢复推理图并直接导出。


