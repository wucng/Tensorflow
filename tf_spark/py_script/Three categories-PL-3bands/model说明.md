# 文档说明

```python
create_pickle_train(image_path, mask_path, img_pixel=9, channels=4)
Function：从图像中提取某一大小的样本做训练数据
Parameter：
image_path 输入影像路径
mask_path 对应的掩膜路径
img_pixel 样本大小
Channels 影像波段数
Return：
样本数据+对应标签

create_pickle_train2(image_path, mask_path, img_pixel=400, channels=3)
与create_pickle_train类似，img_pixel 为400 ，channels 为3

Multiband2Array(path,channels)
Function：将影像转成对应的数组
Parameter：
path 输入影像路径
Channels 影像波段数
Return：
影像对应的数组

next_batch(data, batch_size, flag, img_pixel=3, channels=4)
Function：将数据按批次分割，用于训练
Parameter：
data：数据+标签
batch_size 每次训练的样本数
Flag：bool值 第一次为0，以后为1
img_pixel：取的样本大小
Channels：影像波段数
Return：
图像数据，对应的标签

dense_to_one_hot(labels_dense, num_classes)
Function：将密集标签转成one_hot标签
Parameter：
labels_dense：输入的密集标签
num_classes：分类数
Return：
one_hot标签

超参数
isize：样本大小
img_channel:波段数
img_pixel：样本大小
img_pixel_h：样本高
img_pixel_w：样本宽
training_epochs：迭代次数
batch_size：每次训练的样本数
display_step：打印步数
img_size：样本尺度大小，isize * isize * channels
label_cols：分类数
Dropout：droup out 的保持率
train：模式选取1 tarin ;-1 inference


class Input(object)
数据接口
输入：
image_path（影像路径）
mask_path （对应的掩膜路径）
输出：
data,len(data)

model()
模型建立的函数
输出：
cost,pred_new,optimizer,accuracy

Train(sess)
模型训练
参数：
Sess：传入的Session

Inference2(sess, task_index)
模型推理
参数：
Sess：传入的Session
task_index：指定影像存储位置
```

