参考：

1、https://github.com/aaxwaz/Serving-TensorFlow-Model

2、https://github.com/movchan74/tensorflow_serving_examples

3、https://github.com/zacpang/tensorflow_serving_mnist_example

步骤：

```
cd ~pytest
mkdir test
cd test
vim export_model.py
vim client.py
```
![这里写图片描述](http://img.blog.csdn.net/20171127153454133?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# Train And Export TensorFlow Model

```python
# cd /home/wu/Downloads/serving
# bazel build -c opt ///home/wu/pytest/test:export_model
cd ~pytest/test
python export_model.py --work_dir=../model
```
# Load Exported Model With Standard TensorFlow ModelServer

```python
cd ~/serving
bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server

bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=example1 --model_base_path=/home/wu/pytest/model/
```
# Test The Server

```python
cd ~/serving
bazel build -c opt /home/wu/pytest/test:client

cd ~pytest/test
python client.py --server=localhost:9000
```
