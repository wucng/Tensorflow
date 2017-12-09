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

RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -

RUN apt-get install bazel && apt-get upgrade bazel

RUN pip install grpcio

RUN apt-get clean &&  rm -rf /var/lib/apt/lists/*

RUN pip install tensorflow-serving-api

RUN apt-get remove tensorflow-model-server

RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list

RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -

RUN apt-get install tensorflow-model-server

RUN apt-get upgrade tensorflow-model-server

RUN git clone --recurse-submodules https://github.com/tensorflow/serving
RUN cd serving

RUN cd tensorflow && ./configure && cd ..

RUN bazel build -c opt tensorflow_serving/...
RUN bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server
# 测试
# RUN bazel test -c opt tensorflow_serving/...
