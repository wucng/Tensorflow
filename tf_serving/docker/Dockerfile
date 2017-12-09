# 参考：http://blog.csdn.net/wc781708249/article/details/78594750
# 执行 docker build -t tensorflow-serving:v1 .
FROM ubuntu:16.04
MAINTAINER Mr.wu

RUN apt-get update && apt-get upgrade && \
    apt-get install -y \
	build-essential \
        curl \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
	mlocate \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev \
        libcurl3-dev \
        openjdk-8-jdk\
        openjdk-8-jre-headless \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

Run echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
Run curl https://bazel.build/bazel-release.pub.gpg | apt-key add -

Run apt-get install bazel && apt-get upgrade bazel

Run pip install grpcio

Run apt-get clean &&  rm -rf /var/lib/apt/lists/*

Run pip install tensorflow-serving-api

Run apt-get remove tensorflow-model-server

Run echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list

Run curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -

Run apt-get install tensorflow-model-server

Run apt-get upgrade tensorflow-model-server

Run git clone --recurse-submodules https://github.com/tensorflow/serving
Run cd serving

Run cd tensorflow && ./configure && cd ..

Run bazel build -c opt tensorflow_serving/...
Run bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
# 测试
# Run bazel test -c opt tensorflow_serving/...
