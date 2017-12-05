#!/bin/bash
# chmod +x ./test.sh  修改权限
num=0
while(($num<2181))
do 
    spark-submit \
--master yarn \
--deploy-mode cluster \
--queue default \
--num-executors 45 \
--executor-memory 2G \
--driver-memory 24G \
--archives hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/mnist/mnist.zip#mnist \
--py-files TensorFlowOnSpark/tfspark.zip,myfloder/inference/mnist_dist.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.yarn.executor.memoryOverhead=12288 \
myfloder/inference/mnist_spark.py  \
--images hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/inference \
--labels hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/inference \
--batch_size 40000 \
--format pickle \
--mode inference \
--model hdfs://dm01-08-01.tjidc.dcos.com:8020/user/root/model_arable2 \
--num $num \
--output predictions
    #num=$[num+6]
    #echo $num
    let "num+=6" # 自加6
done 
