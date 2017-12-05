参考：https://tensorflow.google.cn/serving/docker

----------
- [Docker安装](#)
- [哪些容器存在？](#哪些容器存在？)
- [构建一个镜像](#构建一个镜像)
- [运行一个容器](#运行一个容器)
- [构建过程总结](#构建过程总结)


----------
这个目录包含Dockerfiles，使得通过Docker启动和运行TensorFlow服务变得非常容易。

# Docker安装
一般安装说明在[Docker网站上](https://docs.docker.com/engine/installation/)，但是我们在这里给出一些快速链接：

 - OSX：码头工具箱
 - [Ubuntu](https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/)

# 哪些容器存在？

我们目前维护以下Dockerfiles：

[Dockerfile.devel](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel)，这是一个最小的VM，具有构建TensorFlow服务所需的所有依赖关系。

[Dockerfile.devel-gpu](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel-gpu)

# 构建一个镜像

```
docker build --pull -t $USER/tensorflow-serving-devel -f Dockerfile.devel .
```
# 运行一个容器
这假定你已经建立了镜像

Dockerfile.devel：使用开发容器克隆和测试TensorFlow Serving 存储库。

运行容器;

```
docker run -it $USER/tensorflow-serving-devel
```
在正在运行的容器中克隆，配置和测试Tensorflow Serving;

```
git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving/tensorflow
./configure
cd ..
bazel test tensorflow_serving/...
```


# 构建过程总结

```python
mkdir docker
cd docker
vim Dockerfile
# 写入 https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel 中的内容

# 构建镜像
docker build -t 781708249/tf_serving:V1 .
"""
参数说明：
-t ：指定要创建的目标镜像名以及tag
. ：Dockerfile 文件所在目录，也可以指定Dockerfile 的绝对路径
"""

# 查看构建的镜像
docker images / docker image ls

# 使用该镜像运行容器
# 1、运行交互式的容器 缺点 一旦退出 容器就会被关闭
docker run -i -t 781708249/tf_serving:V1 /bin/bash

# 2、后台启动
# docker run -d -P --name V1 781708249/tf_serving:V1
docker run -d -P --name V1 781708249/tf_serving:V1 /bin/sh -c "while true; do echo hello world; sleep 1; done"
docker ps -ls #查看后台运行的容器
docker exec -it 容器ID/容器名 /bin/bash # 进入容器

# 保存容器
docker commit -m="has update" -a="Mr.wu" e218edb10161 781708249/tf_serving:V2
"""
各个参数说明：

-m:提交的描述信息
-a:指定镜像作者
e218edb10161：容器ID （不是镜像ID）
781708249/tf_serving:V2:指定要创建的目标镜像名
"""
# 设置镜像标签
docker images
docker tag 镜像ID 781708249/tf_serving:V3

# 发布镜像
docker login -u 用户名 -p 密码  # 登录到docker hub  
# docker logout 退出
docker push 781708249/tf_serving:V2  # 上传到存储库

# 从远程存储库中提取并运行镜像
docker run -p 4000:80 781708249/tf_serving:V2
```

或者

```python
mkdir docker
cd docker
vim Dockerfile.devel
# 写入 https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel 中的内容

# 构建镜像
docker build --pull -t $USER/tensorflow-serving-devel -f Dockerfile.devel .

# 运行容器
docker run -it $USER/tensorflow-serving-devel

……
……
```
