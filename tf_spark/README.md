TensorflowOnSpark

https://github.com/yahoo/TensorFlowOnSpark


----------
[spark安装](http://note.youdao.com/noteshare?id=d94a999bfe47d69565bbf534e9c63b24&sub=49956E302FD4483B95B8F5A9B00B583D)

[Java环境安装](http://note.youdao.com/noteshare?id=e67f43951d087da41fff6f2a89d3d506&sub=F8F76D03C1F444A7B77B08924F0D6CF5)

[GetStarted_Standalone](http://note.youdao.com/noteshare?id=cde5920f70ff3b90198818c4ca4aa971&sub=DDA97C510B1C4C88B620837068D5C516)

[详解GetStarted_Standalone中每条命令](http://note.youdao.com/noteshare?id=e7d5d36027f15ee3544a48cb4cd7aa27&sub=400FA23EA12E45119DB73C58B3743BD5)

[搭建了GetStarted_Standalone下一次如何重新运行](http://note.youdao.com/noteshare?id=5c8438e4d3f3b29b4d95b68a77225d4b&sub=EE43AE378B7E459EA4C67989D3CF6939)


----------
[GetStarted_YARN安装](http://note.youdao.com/noteshare?id=9ae6faea6fec098a2e6be6d5e12121e3&sub=79606510D055480B9FAA2C685B3D8EB3)

[安装Hadoop，Spark集群模式](http://note.youdao.com/noteshare?id=9074963d546e5a909ac71c072c4bd897&sub=8CED024D1BC74A238B7B8970256BF098)

----------
----------

参考：https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_YARN

----------


# Run MNIST example
## Download/zip the MNIST dataset
```python
# 下载mnist数据
mkdir mnist
cd mnist
curl -O "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
curl -O "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
zip -r mnist.zip * # 创建mnist.zip

# 上传到hdfs
hdfs dfs -mkdir mnist
hdfs dfs -put mnist.zip mnist # 上传到hdfs mnist文件夹下

hdfs dfs -ls mnist # 查看
# hdfs dfs -rm -r mnist # 删除
```

## Convert the MNIST.zip files into HDFS files

```python
# save images and labels as CSV files
spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 45 \
--executor-memory 2G \
--driver-memory 12G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--jars hdfs://xxxx:xx/spark-tensorflow/spark-tensorflow-connector-1.0-SNAPSHOT.jar \
--archives hdfs://xxxx:xx/user/root/mnist/mnist.zip#mnist \
TensorFlowOnSpark/examples/mnist/mnist_data_setup.py \
--output hdfs://xxxx:xx/user/root/mnist/csv \
--format csv

# save images and labels as pickle files
spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 45 \
--executor-memory 2G \
--driver-memory 12G \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--jars hdfs://xxxx:xx/spark-tensorflow/spark-tensorflow-connector-1.0-SNAPSHOT.jar \
--archives hdfs://xxxx:xx/user/root/mnist/mnist.zip#mnist \
TensorFlowOnSpark/examples/mnist/mnist_data_setup.py \
--output hdfs://xxxx:xx/user/root/mnist/pickle \
--format pickle 


```

## Run distributed MNIST training (using feed_dict)

```
${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 4 \
--executor-memory 27G \
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/spark/mnist_dist.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
TensorFlowOnSpark/examples/mnist/spark/mnist_spark.py \
--images hdfs://xxx:xx/user/root/mnist/pickle/train/images \
--labels hdfs://xxx:xx/user/root/mnist/pickle/train/labels \
--mode train \
--model hdfs://xxx:xx/user/root/mnist_model
```

## Run distributed MNIST inference (using feed_dict)
```
${SPARK_HOME}/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 4 \
--executor-memory 27G \
--py-files TensorFlowOnSpark/tfspark.zip,TensorFlowOnSpark/examples/mnist/spark/mnist_dist.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
TensorFlowOnSpark/examples/mnist/spark/mnist_spark.py \
--images hdfs://xxx:xx/user/root/mnist/pickle/test/images \
--labels hdfs://xxx:8020/user/root/mnist/pickle/test/labels \
--mode inference \
--model hdfs://xxx:8020/user/root/mnist_model \
--output hdfs://xxx:8020/user/root/predictions
```

## 附加 hadoop 文件操作命令

```
hdfs dfs -ls # 显示目录
hdfs dfs -ls xxx/|wc -l # 显示xxx目录下的文件和文件夹个数
hdfs dfs -mkdir xxx # 新建目录
hdfs dfs -rm -r xxx # 删除文件或目录
hdfs dfs -put  xxx data # 将xxx 上传到 hdfs的data目录
hdfs dfs -get xxx ./ # 将hdfs的xxx（文件或文件夹）复制到本地

yarn application -kill application_1502181070712_0574  # 杀掉进程

spark-submit test.py  # 执行脚本 test.py
```
