参考：

1、https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker

2、https://tensorflow.google.cn/install/install_linux#installing_with_docker


----------

按照以下步骤通过Docker安装TensorFlow：

1、按照[Docker文档](https://docs.docker.com/engine/installation/)中的描述在您的机器上安装Docker。

2、或者，创建一个名为docker的Linux组，以允许启动没有sudo的容器，如[Docker文档](https://docs.docker.com/engine/installation/linux/linux-postinstall/)中所述。 （如果你不这样做，每次调用Docker时都必须使用sudo。）

3、要安装支持GPU的TensorFlow版本，您必须先安装存储在github中的[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)。

4、启动包含[TensorFlow二进制镜像](https://hub.docker.com/r/tensorflow/tensorflow/tags/)之一的Docker容器。

本节的其余部分将介绍如何启动Docker容器。

# CPU-only
要启动一个仅支持CPU的Docker容器（即没有GPU支持），请输入以下格式的命令：

```
$ docker run -it -p hostPort:containerPort TensorFlowCPUImage
```

 - -p hostPort：containerPort是可选的。 如果您打算从shell运行TensorFlow程序，请省略此选项。 如果您计划将TensorFlow程序作为Jupyter笔记本运行，请将hostPort和containerPort设置为8888.如果您想在容器内运行TensorBoard，请添加第二个-p标志，将hostPort和containerPort设置为6006。
 - TensorFlowCPUImage是必需的。 它标识了Docker容器。 指定下列值之一：
	 - gcr.io/tensorflow/tensorflow，这是TensorFlow CPU binary image。
	 - gcr.io/tensorflow/tensorflow:latest-devel，这是最新的TensorFlow CPU二进制镜像加源代码。
	 - gcr.io/tensorflow/tensorflow:version，它是TensorFlow CPU二进制镜像的指定版本（例如，1.1.0rc1）。
	 - gcr.io/tensorflow/tensorflow:version-devel，它是TensorFlow GPU二进制镜像的源代码的指定版本（例如，1.1.0rc1）。

gcr.io是Google容器注册表。 请注意，[docker hub](https://hub.docker.com/r/tensorflow/tensorflow/)上也提供了一些TensorFlow镜像。

例如，以下命令在Docker容器中启动最新的TensorFlow CPU二进制镜像，您可以在其中运行TensorFlow程序：

```
# shell中打开
$ docker run -it gcr.io/tensorflow/tensorflow bash
# 或
$ docker run --name test -it tensorflow/tensorflow /bin/bash
```
以下命令还会在Docker容器中启动最新的TensorFlow CPU二进制镜像。 但是，在这个Docker容器中，您可以在Jupyter笔记本中运行TensorFlow程序：

```python
# 主机上打开网站链接到Jupyter，参看下图
$ docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow
# 或
$ docker run --name test -it -p 8888:8888 tensorflow/tensorflow

……
……
Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=c15b9bf78a1847173f29383895b6ffc86cacd0a59c98a211
# 在主机上打开该网站即可
```

![这里写图片描述](http://img.blog.csdn.net/20171213083641650?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

Docker首次启动时会下载TensorFlow二进制镜像。

或

```python
# 先下载镜像
docker pull tensorflow/tensorflow:latest
# 再启动容器
docker run --name test -it -p 8888:8888 tensorflow/tensorflow
# 或
docker run --name test -it tensorflow/tensorflow /bin/bash
```

----------
# GPU support
在支持GPU的情况下安装TensorFlow之前，请确保您的系统符合所有[NVIDIA软件要求](https://tensorflow.google.cn/install/install_linux#NVIDIARequirements)。 要启动具有NVidia GPU支持的Docker容器，请输入以下格式的命令：

```
$ nvidia-docker run -it -p hostPort:containerPort TensorFlowGPUImage
```

 - -p hostPort：containerPort是可选的。如果您打算从shell运行TensorFlow程序，请省略此选项。如果您计划将TensorFlow程序作为Jupyter笔记本运行，请将hostPort和containerPort设置为8888。
 - TensorFlowGPUImage指定Docker容器。您必须指定下列值之一：
	 - gcr.io/tensorflow/tensorflow:latest-gpu，这是最新的TensorFlow GPU二进制图像。
	 - gcr.io/tensorflow/tensorflow:latest-devel-gpu，这是最新的TensorFlow
   GPU二进制图像加源代码。
	 - gcr.io/tensorflow/tensorflow:version-gpu，这是TensorFlow
   GPU二进制映像的指定版本（例如0.12.1）。
	 - gcr.io/tensorflow/tensorflow:version-devel-gpu，它是TensorFlow
   GPU二进制图像的源代码的指定版本（例如0.12.1）。

我们建议安装其中一个最新版本。例如，以下命令在Docker容器中启动最新的TensorFlow GPU二进制映像，您可以在其中运行TensorFlow程序：

附加：
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
[nvidia hub](https://hub.docker.com/r/nvidia/cuda/tags/)镜像

确保你已经安装了[NVIDIA驱动程序](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)和[受支持](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#which-docker-packages-are-supported)的Docker版本（[请参阅先决条件](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-%28version-2.0%29#prerequisites)）。

<font color=#d000 size=5>注：</font>主机安装nvidia驱动和cuda、cudann 参考[这里](http://blog.csdn.net/wc781708249/article/details/77989339)
docker安装参考[这里](https://docs.docker.com/engine/installation/)
```python
# 安装nvidia-docker

# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
sudo apt-get purge -y nvidia-docker

# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

# Test nvidia-smi with the latest official CUDA image
# docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
# 下载cuda 镜像
# docker pull nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
```

```python
# shell中打开
$ nvidia-docker run -it gcr.io/tensorflow/tensorflow:latest-gpu bash
# 或
$ nvidia-docker run -it tensorflow/tensorflow:latest-gpu /bin/bash
```
以下命令还会在Docker容器中启动最新的TensorFlow GPU二进制镜像。 在这个Docker容器中，您可以在Jupyter笔记本中运行TensorFlow程序：

```python
# 主机网页上通过Jupyter打开
$ nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu
# 或
$ nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu
```
以下命令将安装较旧的TensorFlow版本（0.12.1）：

```python
$ nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:0.12.1-gpu
# 或
$ nvidia-docker run --name test -it -p 8888:8888 tensorflow/tensorflow:0.12.1-gpu
```
Docker首次启动时会下载TensorFlow二进制镜像。 有关更多详细信息，请参阅[TensorFlow docker自述文件](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker)。
