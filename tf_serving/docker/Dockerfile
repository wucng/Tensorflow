FROM ubuntu:16.04
MAINTAINER Mr.wu

# ENV HOME /root

RUN apt-get update && \
    apt-get upgrade && \
    apt-get install openjdk-8-jdk && \
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add - && \
    apt-get update && sudo apt-get install bazel && \
    apt-get upgrade bazel && \
    pip install grpcio && \
    apt-get install -y \
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
        zlib1g-dev && \
    pip install tensorflow-serving-api && \
    apt-get remove tensorflow-model-server && \
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
    curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add - && \
    apt-get update && apt-get install tensorflow-model-server && \
    apt-get upgrade tensorflow-model-server

RUN cd ~
RUN git clone --recurse-submodules https://github.com/tensorflow/serving && \
cd serving && \
cd tensorflow && \
./configure && \
cd .. && \
bazel build -c opt tensorflow_serving/... && \
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server && \
bazel test -c opt tensorflow_serving/...
