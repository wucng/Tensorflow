注意事项：

1、在每台机器上都安装tensorflow （版本保持一致） 安装tensorflow时默认会安装grpc

2、将mnist_dist.py复制到每台机器上 （也可以将mnis数据也复制到每台机器上 省的下载）

3、按照 执行方式中的命令  在对应的机器上执行（ip与任务索引对应） 可以写成shell脚本来执行

4、由于ps节点不会自动关闭，需强制关闭，必须重新连接下所有机器（如：关闭终端 重新打开终端） 再次执行集群才不会报错

5、如果模型保存在本地路径，需将worker：0 节点上的模型参数复制到 ps 节点，才能进行模型再训练；如果模型保存在hdfs上，则不需要

参考：[\[集群版\]tf read hdfs.py](https://github.com/fengzhongyouxia/Tensorflow/blob/master/tf_spark/py_script/%5B%E9%9B%86%E7%BE%A4%E7%89%88%5Dtf%20read%20hdfs.py)




执行方式：

```python
python3 mnist_dist.py --ps_hosts=192.168.146.137:2220 --worker_hosts=192.168.146.133:2221,192.168.146.136:2222 --job_name="ps" --task_index=0
python3 mnist_dist.py --ps_hosts=192.168.146.137:2220 --worker_hosts=192.168.146.133:2221,192.168.146.136:2222 --job_name="worker" --task_index=0
python3 mnist_dist.py --ps_hosts=192.168.146.137:2220 --worker_hosts=192.168.146.133:2221,192.168.146.136:2222 --job_name="worker" --task_index=1
```


```python
python3 mnist_replica.py --ps_hosts=192.168.146.137:2222 --worker_hosts=192.168.146.133:2223,192.168.146.136:2224 --job_name="ps" --task_index=0
python3 mnist_replica.py --ps_hosts=192.168.146.137:2222 --worker_hosts=192.168.146.133:2223,192.168.146.136:2224 --job_name="worker" --task_index=0
python3 mnist_replica.py --ps_hosts=192.168.146.137:2222 --worker_hosts=192.168.146.133:2223,192.168.146.136:2224 --job_name="worker" --task_index=1
```

```python
python3 mnist.py --ps_hosts=192.168.146.137:2221 --worker_hosts=192.168.146.133:2222,192.168.146.136:2223 --job_name="ps" --task_index=0
python3 mnist.py --ps_hosts=192.168.146.137:2221 --worker_hosts=192.168.146.133:2222,192.168.146.136:2223 --job_name="worker" --task_index=0
python3 mnist.py --ps_hosts=192.168.146.137:2221 --worker_hosts=192.168.146.133:2222,192.168.146.136:2223 --job_name="worker" --task_index=1
```
