参考：
1、https://tensorflow.google.cn/serving/serving_inception
2、https://tensorflow.google.cn/serving/serving_basic


----------
主机安装tensorflow serving 参考[这里](http://blog.csdn.net/wc781708249/article/details/78594750)
主机使用tensorflow serving部署mnist模型参考[这里](http://blog.csdn.net/wc781708249/article/details/78596459)
Docker安装tensorflow serving 参考[这里](http://blog.csdn.net/wc781708249/article/details/78722958)
Docker中部署Inception模型 参考[这里](http://blog.csdn.net/wc781708249/article/details/78781492)


----------
# 1、创建一个Docker镜像
参考：Docker安装tensorflow serving 参考[这里](http://blog.csdn.net/wc781708249/article/details/78722958)
## 运行容器
```python
docker pull registry.cn-hangzhou.aliyuncs.com/781708249/tensorflow-serving:v1 # 已经配置好的tensorflow serving 从阿里镜像拉下来
git clone --recurse-submodules https://github.com/tensorflow/serving # serving下载到主机上
docker run --name=mnist_container -it -v /home/wu/serving:/serving registry.cn-hangzhou.aliyuncs.com/781708249/tensorflow-serving:v1 /bin/bash # 使用-v 挂载到容器中
```
## 配置和构建TensorFlow服务

```
root@c97d8e820ced:/# cd serving/tensorflow
root@c97d8e820ced:/serving/tensorflow# ./configure
root@c97d8e820ced:/serving# cd ..
root@c97d8e820ced:/serving# bazel build -c opt tensorflow_serving/example/...
```

```
root@c97d8e820ced:/serving# bazel build -c opt tensorflow_serving/model_servers:tensorflow_model_server
```
## 在容器中导出初始模型
在正在运行的容器中，我们运行[mnist_saved_model.py](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py)
```
root@c97d8e820ced:/serving# rm -rf /tmp/mnist_model
root@c97d8e820ced:/serving# bazel-bin/tensorflow_serving/example/mnist_saved_model /tmp/mnist_model
root@c97d8e820ced:/serving# [Ctrl-p] + [Ctrl-q]
```
## 提交镜像进行部署

```
$ docker commit mnist_container $USER/mnist_serving
$ docker stop mnist_container 
```

# 2、在本地Docker容器中运行
我们使用构建的镜像在本地测试服务工作流程。

```python
# $ docker run -it $USER/mnist_serving
$ docker run -it -v /home/wu/serving:/serving $USER/mnist_serving
```
## 启动服务器
在容器中运行[gRPC](https://grpc.io/) tensorflow_model_server

```
root@f07eec53fd95:/# cd serving
root@f07eec53fd95:/serving# bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/ &> mnist_log &
[2] 80
```
## 查询服务器
使用[mnist_client.py](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_client.py)查询服务器。 

```
root@f07eec53fd95:/serving# bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000

Extracting /tmp/train-images-idx3-ubyte.gz
Extracting /tmp/train-labels-idx1-ubyte.gz
Extracting /tmp/t10k-images-idx3-ubyte.gz
Extracting /tmp/t10k-labels-idx1-ubyte.gz
........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
Inference error rate: 10.4%
```
mnist模型部署成功！


----------


进一步使用参考：

1、https://github.com/fengzhongyouxia/Tensorflow/tree/master/tf_serving/MNIST

2、https://github.com/fengzhongyouxia/Tensorflow/tree/master/tf_serving/1
