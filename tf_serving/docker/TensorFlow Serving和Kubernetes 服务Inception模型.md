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

```
root@c97d8e820ced:/serving# curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
root@c97d8e820ced:/serving# tar xzf inception-v3-2016-03-01.tar.gz
root@c97d8e820ced:/serving# ls inception-v3
README.txt  checkpoint  model.ckpt-157585
root@c97d8e820ced:/serving# bazel-bin/tensorflow_serving/example/inception_saved_model --checkpoint_dir=inception-v3 --output_dir=/tmp/inception-export
Successfully loaded model from inception-v3/model.ckpt-157585 at step=157585.
Successfully exported model to /tmp/inception-export
root@c97d8e820ced:/serving# ls /tmp/inception-export
1
root@c97d8e820ced:/serving# [Ctrl-p] + [Ctrl-q]
```
## 提交镜像进行部署
请注意，我们在上面的说明结束时从容器中分离出来，而不是终止它，因为我们要[commit](https://docs.docker.com/engine/reference/commandline/commit/)对用于Kubernetes部署的新镜像$ USER / inception_serving的所有更改。

```
$ docker commit inception_container $USER/inception_serving
$ docker stop inception_container
```
# 第1部分：在本地Docker容器中运行
我们使用构建的镜像在本地测试服务工作流程。

```
$ docker run -it $USER/inception_serving
```

## 启动服务器

在容器中运行[gRPC](https://grpc.io/) tensorflow_model_server。

```
root@f07eec53fd95:/# cd serving
root@f07eec53fd95:/serving# bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=inception --model_base_path=/tmp/inception-export &> inception_log &
[1] 45
```
## 查询服务器
使用[inception_client.py](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_client.py)查询服务器。 客户端通过gRPC向服务器发送由命令行参数指定的图像，以便分类成对于[ImageNet](http://www.image-net.org/)类别的人类可读描述。

```
root@f07eec53fd95:/serving# bazel-bin/tensorflow_serving/example/inception_client --server=localhost:9000 --image=/path/to/my_cat_image.jpg
outputs {
  key: "classes"
  value {
    dtype: DT_STRING
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    string_val: "tiger cat"
    string_val: "Egyptian cat"
    string_val: "tabby, tabby cat"
    string_val: "lynx, catamount"
    string_val: "Cardigan, Cardigan Welsh corgi"
  }
}
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    float_val: 9.5486907959
    float_val: 8.52025032043
    float_val: 8.05995368958
    float_val: 4.30645561218
    float_val: 3.93207240105
  }
}

root@f07eec53fd95:/serving# exit
```

有用！ 服务器成功分类你的猫的形象！

# 第2部分：在Kubernetes中部署

在本节中，我们使用第0部分中构建的容器镜像像在[Google云端平台](http://cloud.google.com/)中部署带有[Kubernetes](https://kubernetes.io/)的服务集群。

## Cloud项目登录
在这里，我们假定您已经创建并登录了一个名为tensorflow-serving的[gcloud](https://cloud.google.com/sdk/gcloud/)项目。

```
gcloud auth login --project tensorflow-serving
```
## 创建一个容器集群
首先，我们创建一个[Google Container Engine](https://cloud.google.com/container-engine/)集群来进行服务部署。

```
$ gcloud container clusters create inception-serving-cluster --num-nodes 5
Creating cluster inception-serving-cluster...done.
Created [https://container.googleapis.com/v1/projects/tensorflow-serving/zones/us-central1-f/clusters/inception-serving-cluster].
kubeconfig entry generated for inception-serving-cluster.
NAME                       ZONE           MASTER_VERSION  MASTER_IP        MACHINE_TYPE   NODE_VERSION  NUM_NODES  STATUS
inception-serving-cluster  us-central1-f  1.1.8           104.197.163.119  n1-standard-1  1.1.8         5          RUNNING
```

设置gcloud容器命令的默认群集并将群集凭据传递给[kubectl](https://kubernetes.io/docs/reference/kubectl/overview/)。

```
$ gcloud config set container/cluster inception-serving-cluster
$ gcloud container clusters get-credentials inception-serving-cluster
Fetching cluster endpoint and auth data.
kubeconfig entry generated for inception-serving-cluster.
```
## 上传Docker镜像
现在让我们将图片推送到[Google Container Registry](https://cloud.google.com/container-registry/docs/)，以便我们可以在Google Cloud Platform上运行它。

首先我们使用Container Registry格式和我们的项目名称来标记$ USER / inception_serving图像，

```
$ docker tag $USER/inception_serving gcr.io/tensorflow-serving/inception
```
接下来，我们将图像推送到注册表，

```
$ gcloud docker -- push gcr.io/tensorflow-serving/inception
```
## 创建Kubernetes部署和服务
部署包含3个由[Kubernetes部署控制](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)的inception_inference服务器副本。 [Kubernetes Service](https://kubernetes.io/docs/concepts/services-networking/service/)与[External Load Balancer](https://kubernetes.io/docs/tasks/access-application-cluster/create-external-load-balancer/)一起向外部复制副本。

我们使用示例[Kubernetes config inception_k8s.yaml](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_k8s.yaml)创建它们。

```
$ kubectl create -f tensorflow_serving/example/inception_k8s.yaml
deployment "inception-deployment" created
service "inception-service" created
```
要查看部署和窗格的状态，请执行以下操作：

```
$ kubectl get deployments
NAME                    DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
inception-deployment    3         3         3            3           5s
```

```
$ kubectl get pods
NAME                         READY     STATUS    RESTARTS   AGE
inception-deployment-bbcbc   1/1       Running   0          10s
inception-deployment-cj6l2   1/1       Running   0          10s
inception-deployment-t1uep   1/1       Running   0          10s
```
要查看服务的状态：

```
$ kubectl get services
NAME                    CLUSTER-IP       EXTERNAL-IP       PORT(S)     AGE
inception-service       10.239.240.227   104.155.184.157   9000/TCP    1m
```
可能需要一段时间才能启动并运行。

```
$ kubectl describe service inception-service
Name:           inception-service
Namespace:      default
Labels:         run=inception-service
Selector:       run=inception-service
Type:           LoadBalancer
IP:         10.239.240.227
LoadBalancer Ingress:   104.155.184.157
Port:           <unset> 9000/TCP
NodePort:       <unset> 30334/TCP
Endpoints:      <none>
Session Affinity:   None
Events:
  FirstSeen LastSeen    Count   From            SubobjectPath   Type        Reason      Message
  --------- --------    -----   ----            -------------   --------    ------      -------
  1m        1m      1   {service-controller }           Normal      CreatingLoadBalancer    Creating load balancer
  1m        1m      1   {service-controller }           Normal      CreatedLoadBalancer Created load balancer
```
服务外部IP地址列在LoadBalancer Ingress旁边。

## 查询模型
我们现在可以从本地主机的外部地址查询服务。

```
$ bazel-bin/tensorflow_serving/example/inception_client --server=104.155.184.157:9000 --image=/path/to/my_cat_image.jpg
outputs {
  key: "classes"
  value {
    dtype: DT_STRING
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    string_val: "tiger cat"
    string_val: "Egyptian cat"
    string_val: "tabby, tabby cat"
    string_val: "lynx, catamount"
    string_val: "Cardigan, Cardigan Welsh corgi"
  }
}
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    float_val: 9.5486907959
    float_val: 8.52025032043
    float_val: 8.05995368958
    float_val: 4.30645561218
    float_val: 3.93207240105
  }
}
```
您已成功部署在Kubernetes服务的先启模式！
