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
$ docker run -it gcr.io/tensorflow/tensorflow bash
```
以下命令还会在Docker容器中启动最新的TensorFlow CPU二进制镜像。 但是，在这个Docker容器中，您可以在Jupyter笔记本中运行TensorFlow程序：

```
$ docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow
```
Docker首次启动时会下载TensorFlow二进制镜像。
