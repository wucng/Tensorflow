执行方式

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
