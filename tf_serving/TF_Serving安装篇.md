参考：
1、https://tensorflow.google.cn/
2、https://www.tensorflow.org/
3、https://tensorflow.google.cn/serving/setup
4、
http://note.youdao.com/noteshare?id=92af08c5ceff28db9748f822fee05322&sub=7E5B724B2443405C9DD3D8E22C3A1EE9


----------
环境Ubuntu 16.04 LTS

<font color=#d000 size=5>注：</font> tf_serving 默认是python2版，python3可以参考该安装方法

----------

[toc]

----------


# 1、安装Bazel
TensorFlow Serving需要Bazel 0.5.4或更高。 您可以在[这里](https://docs.bazel.build/versions/master/install.html)找到Bazel安装说明。
ubuntu安装参考：https://docs.bazel.build/versions/master/install-ubuntu.html 
<font color=#d000 size=5>注：</font>只支持ubuntu 16.04 (LTS) ，  ubuntu 14.04 (LTS)


----------


## 方法一  Using Bazel custom APT repository (推荐使用)

### 1、Install JDK 8

```
sudo apt-get install openjdk-8-jdk
```
On <font color=#d000 size=5>Ubuntu 14.04 LTS </font>you'll have to use a PPA:(Ubuntu 16.04 LTS 跳过)

```
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update && sudo apt-get install oracle-java8-installer
```
### 2、Add Bazel distribution URI as a package source (one time setup)

```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
```
如果要安装Bazel的测试版本，请将stable替换为 testing。

### 3、Install and update Bazel

```
sudo apt-get update && sudo apt-get install bazel
```
一旦安装，您可以升级到更新版本的Bazel：

```
sudo apt-get upgrade bazel
```


----------
## 方法二  Install using binary installer
二进制安装程序位于Bazel的[GitHub发布页面上](https://github.com/bazelbuild/bazel/releases)。

### 1、Install required packages

```
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python
```
### 2、Download Bazel
注意：在本文档中列出的安装程序文件名称中，用适当的Bazel版本号替换<version>。
Go to Bazel's [GitHub releases page](https://github.com/bazelbuild/bazel/releases).

下载二进制安装程序`bazel-<version>-installer-linux-x86_64.sh`。 该安装程序包含Bazel二进制文件和所需的JDK，即使已经安装了JDK，也可以使用该安装程序。

请注意，`bazel-<version>-without-jdk-installer-linux-x86_64.sh`也存在。 这是一个没有嵌入式JDK 8的版本。如果你已经安装了JDK 8，只能使用这个安装程序。

### 3、Run the installer

```
chmod +x bazel-<version>-installer-linux-x86_64.sh
./bazel-<version>-installer-linux-x86_64.sh --user
```
--user 标志将Bazel安装到系统上的`$HOME/bin`目录中，并将.bazelrc路径设置为`$HOME/.bazelrc`。 使用--help命令查看其他安装选项。

### 4、Set up your environment
如果使用上面的`--user`标志运行Bazel安装程序，则Bazel可执行文件将安装在您的`$HOME/bin`目录中。 将此目录添加到默认路径是一个好主意，如下所示：

```
export PATH="$PATH:$HOME/bin"
```
您也可以将此命令添加到`~/.bashrc`文件中。


----------


## 方法三 Compile Bazel from source
参考：https://docs.bazel.build/versions/master/install-compile-source.html


----------
# 2、安装gRPC
我们的教程使用[gRPC](https://grpc.io/)（1.0.0或更高版本）作为我们的RPC框架。 你可以在[这里](https://github.com/grpc/grpc/tree/master/src/python/grpcio)找到安装说明。
参考：https://github.com/grpc/grpc/tree/master/src/python/grpcio

## 方法一  From PyPI

```python
# 本地安装
pip install grpcio

# Ubuntu安装
sudo pip install grpcio  # python2

# Windows安装，需先安装pip.exe组件
pip.exe install grpcio
```

## 方法二 From Source
从源代码构建需要你有Python头文件（通常是一个名为`python-dev`的包）。

```
$ export REPO_ROOT=grpc  # REPO_ROOT can be any directory of your choice
$ git clone -b $(curl -L https://grpc.io/release) https://github.com/grpc/grpc $REPO_ROOT
$ cd $REPO_ROOT
$ git submodule update --init

# For the next two commands do `sudo pip install` if you get permission-denied errors
$ pip install -rrequirements.txt
$ GRPC_PYTHON_BUILD_WITH_CYTHON=1 pip install .
```
您目前无法在Windows上从源代码安装Python。 在MSYS2中可能会出现一些问题（按照Linux的说明），但目前尚未得到官方的支持。

##Troubleshooting
参考：https://github.com/grpc/grpc/tree/master/src/python/grpcio

# 3、安装Packages
要安装TensorFlow server依赖关系，请执行以下操作：

```
sudo apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        libcurl3-dev \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev
```
构建TensorFlow所需的软件包列表会随着时间而改变，所以如果遇到任何问题，请参阅TensorFlow的[build instructions](https://tensorflow.google.cn/install/install_sources)。 请特别注意可能需要运行的`apt-get install`和`pip install`命令。

----------


# 4、(options)安装TensorFlow Serving Python API PIP package

要运行Python客户端代码而无需安装Bazel，可以使用以下命令来安装tensorflow-serving-api PIP包：

```python
# tensorflow serving 默认是python2版
pip install tensorflow-serving-api # 貌似只能安装python2版
pip3 install tensorflow-serving-api # 安装不成功
```


----------


# 5、安装TensorFlow Serving
## (options)使用apt-get安装

### Available binaries
TensorFlow Serving ModelServer二进制文件有两种版本：

`tensorflow-model-server`：完全优化的服务器，使用一些平台特定的编译器优化，如SSE4和AVX指令。 这应该是大多数用户的首选选项，但可能无法在一些较旧的机器上工作。

`tensorflow-model-server-universal`：编译基本的优化，但不包括特定于平台的指令集，所以应该在大多数机器上工作，如果不是所有的机器。 如果tensorflow-model-server不适合你，请使用这个。 请注意，这两个软件包的二进制名称是相同的，所以如果您已经安装tensorflow-model-server，您应该首先使用卸载

```
sudo apt-get remove tensorflow-model-server
```
### 安装ModelServer
1、添加TensorFlow Serving 分发URI作为包源（一次性设置）

```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list

curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
```
2、安装并更新TensorFlow ModelServer

```
sudo apt-get update && sudo apt-get install tensorflow-model-server
```
安装完成后，可以使用`tensorflow_model_server`命令调用二进制文件。
您可以升级到更新版本的`tensorflow-model-server`：

```
sudo apt-get upgrade tensorflow-model-server
```
注：在上述命令中，如果处理器不支持AVX指令，请将`tensorflow-model-server`替换为`tensorflow-model-server-universal`。


----------


## 源码安装
### 方法一（非docker使用该方法）
#### Clone the TensorFlow Serving repository

```
git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving
```
--recurse-submodules需要获取TensorFlow serving所依赖的TensorFlow，gRPC和其他库。 请注意，这些说明将安装TensorFlow Serving的最新主分支。 如果你想安装一个特定的分支（比如一个release 分支），把`-b <branchname>`传给git clone命令。

#### Install prerequisites
按照上面的先决条件部分安装所有依赖项。 配置TensorFlow，运行

```python
cd tensorflow
./configure
cd ..
```
如果您在设置TensorFlow或其依赖项时遇到任何问题，请参阅[TensorFlow安装](https://tensorflow.google.cn/install/)说明。

#### Build
TensorFlow Serving 使用Bazel build。 使用Bazel命令来构建单个目标或整个源代码树。

要构建整个树，请执行：

```
bazel build -c opt tensorflow_serving/...
```
二进制文件放在bazel-bin目录下，可以使用如下命令运行：

```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
```
要测试您的安装，请执行：

```
bazel test -c opt tensorflow_serving/...
```
有关运行TensorFlow服务的更深入示例，请参阅[基本教程](https://tensorflow.google.cn/serving/serving_basic)和[高级教程](https://tensorflow.google.cn/serving/serving_advanced)。

####（options）Optimized build
可以使用一些特定于平台的指令集（例如AVX）进行编译，这可以显着提高性能。 无论您在文档中看到“bazel build”，都可以添加标志-c opt --copt = -msse4.1 --copt = -msse4.2 --copt = -mavx --copt = -mavx2 --copt = -mfma --copt = -O3（或这些标志的一些子集）。 例如：

```
bazel build -c opt --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-O3 tensorflow_serving/...
```
注意：这些指令集在所有机器上都不可用，特别是对于较旧的处理器，所以它可能不适用于所有的标志。 你可以尝试一些子集，或者恢复到保证在所有机器上工作的基本“-c opt”。


----------
### 方法二（针对docker版本）
参考：http://geek.csdn.net/news/detail/194233
#### Continuous integration build
我们使用TensorFlow [ci_build](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/ci_build)基础架构进行[continuous integration build](http://ci.tensorflow.org/view/Serving/job/serving-master-cpu/)，为您提供使用docker的简化开发。 所有你需要的是git和docker。 无需手动安装所有其他依赖项。

```
git clone --recursive https://github.com/tensorflow/serving
cd serving
CI_TENSORFLOW_SUBMODULE_PATH=tensorflow tensorflow/tensorflow/tools/ci_build/ci_build.sh CPU bazel test //tensorflow_serving/...
# GPU版本 将上面的CPU 改成 GPU
```
注意：serving 目录映射到容器中。 你可以在Docker容器之外进行开发（在你最喜欢的编辑器中），当你运行这个构建时，它将随着你的改变而建立。

----------



