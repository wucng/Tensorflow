# -*- coding: UTF-8 -*-

"""
针对多于4个波段的影像
使用GDAL读取影像
存储成pickle
"""
import m1
import numpy as np
import pickle
import os
import zlib
import gzip

'''
# dir_name 为文件路径
# img_pixel 为图像的像素尺寸
'''
def create_pickle_train(dir_name,img_pixel=60,channels=4):
    flag=False
    for _,dirs,_ in os.walk(dir_name):
        for filename in dirs: # 文件夹名 取文件名作为标签
            file_path=os.path.join(dir_name,filename) # 文件夹路径
            # for _ , _,img in os.walk(file_path):
            for img_name in os.listdir(file_path): # 文件夹下的文件名
                imgae_path = os.path.join(file_path, img_name) # 文件路径

                img=m1.Multiband2Array(imgae_path) # 使用GDAL方法读取影像 可以读取多于3个波段的影像

                data1=img.reshape((-1,img_pixel*img_pixel*channels)) # 展成一行
                label=np.array([int(filename)]) # 文件名作为标签

                data2=np.append(data1,label)[np.newaxis,:] # 数据+标签

                # data2=data2.tostring()  # 转成byte，缩小文件大小
                # data2=zlib.compress(data2) # 使用zlib将数据进一步压缩


                if flag==False:
                    data=data2
                if flag==True:
                    data=np.vstack((data,data2))  # 上下合并
                flag = True
    return data
    # with gzip.open(dir_name + '/train_data.pkl', 'wb') as writer: # 以压缩包方式创建文件，进一步压缩文件
    #     pickle.dump(data, writer)  # 数据存储成pickle文件


'''
# dir_name 为文件路径
# img_pixel 为图像的像素尺寸
# img_names 记录图像名 测试数据时记录文件名
'''
def create_pickle_test(dir_name,img_pixel=60,channels=4,img_names=[]):
    flag = False
    for _, dirs, _ in os.walk(dir_name):
        for filename in dirs:  # 文件夹名 取文件名作为标签
            file_path = os.path.join(dir_name, filename)  # 文件夹路径
            # for _ , _,img in os.walk(file_path):
            for img_name in os.listdir(file_path):  # 文件夹下的文件名

                # img_names.append(img_name)  # 依次记录图像名

                imgae_path = os.path.join(file_path, img_name)  # 文件路径

                img = m1.Multiband2Array(imgae_path)  # 使用GDAL方法读取影像 可以读取多于3个波段的影像

                data1 = img.reshape((-1, img_pixel * img_pixel * channels))  # 展成一行
                label = np.array([int(filename)])  # 文件名作为标签

                data2 = np.append(data1, label)[np.newaxis, :]  # 数据+标签

                data2 = data2.tostring()  # 转成byte，缩小文件大小
                data2 = zlib.compress(data2)  # 使用zlib将数据进一步压缩

                if flag == False:
                    data = data2
                if flag == True:
                    data = np.vstack((data, data2))  # 上下合并
                flag = True

    with gzip.open(dir_name + 'test_data.pkl', 'wb') as writer:  # 以压缩包方式创建文件，进一步压缩文件
        pickle.dump(data, writer)  # 数据存储成pickle文件

    # with open(dir_name + 'images_name.pkl', 'wb') as output:
    #     pickle.dump(img_names, output)  # 将图像名记录列表存储成pickle文件


'''
# filename 为生成的pickle 路径名
# img_pixel 为图像像素尺寸
# channels 为波段数
# return： 特征+标签
'''
def read_and_decode(filename,img_pixel=60,channels=4):

    with gzip.open(filename, 'rb') as pkl_file: # 打开文件
        data = pickle.load(pkl_file) # 加载数据

    for i in range((data.shape)[0]):
        data_1 = np.fromstring(zlib.decompress(data[i, :]))# 转码 先解压缩，再转成数组
        if i == 0:
            data_2 = data_1
        else:
            data_2 = np.vstack((data_2, data_1)) # 上下合并

    # data= data_2.reshape((-1, img_pixel * img_pixel * channels + 1))  # 数据每一行都是 图像+标签

    # 将数据按行打乱 这里不打乱 否则测试时 数据与图像名对应不起来
    # index = [i for i in range(len(data_2))]  # len(data_2)得到的行数
    # np.random.shuffle(index)  # 将索引打乱
    # data_2 = data_2[index]

    return data_2


'''
其他工具
'''
# ---------------生成多列标签 如：0,1 对应为[1,0],[0,1]------------#
# 单列标签转成多列标签
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  # 从标量类标签转换为一个one-hot向量
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  # print index_offset
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def dense_to_one_hot2(labels_dense,num_classes):
    labels_dense=np.array(labels_dense,dtype=np.uint8)
    num_labels = labels_dense.shape[0] # 标签个数
    labels_one_hot=np.zeros((num_labels,num_classes),np.uint8)
    for i,itenm in enumerate(labels_dense):
        labels_one_hot[i,itenm]=1
        # 如果labels_dense不是int类型，itenm就不是int，此时做数组的切片索引就会报错，
        # 数组索引值必须是int类型，也可以 int(itenm) 强制转成int
        # labels_one_hot[i, :][itenm] = 1
    return labels_one_hot

'''
def parse(ln):
  lbl, img = ln.split('|')
  image = [int(x) for x in img.split(',')]
  label = numpy.zeros(10)
  label[int(lbl)] = 1.0
  return (image,label)
'''

# ------------next_batch------------#
'''
注：
每次 data传入next_batch()完成，进行下一次传入时，先进行打乱
如下面的做法：

total_batch = int(img_nums / batch_size)
data=read_and_decode(filename,img_pixel=60,channels=3)

for epoch in range(training_epochs):
    # 将数据按行打乱
    index = [i for i in range(len(data))]  # len(data)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    data = data[index]
    for i in range(total_batch):
        img, label=next_batch(data,batch_size,img_pixel=60,channels=4)
        ......
'''
# 按batch_size提取数据
# batch_size为每次批处理样本数
# data包含特征+标签 每一行都是 特征+标签
start_index=0
def next_batch(data,batch_size,img_pixel=60,channels=4):
    global start_index  # 必须定义成全局变量
    global second_index  # 必须定义成全局变量

    second_index=start_index+batch_size
    if second_index>len(data):
        second_index=len(data)
    data1=data[start_index:second_index]
    # lab=labels[start_index:second_index]
    start_index=second_index
    if start_index>=len(data):
        start_index = 0

    # 将每次得到batch_size个数据按行打乱
    index = [i for i in range(len(data1))]  # len(data1)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    data1 = data1[index]

    # 提起出数据和标签
    img = data1[:, 0:img_pixel * img_pixel * channels]

    # img = img * (1. / img.max) - 0.5
    img = img * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
    # img=img.astype(float) # 类型转换

    label = data1[:, -1]
    label = label.astype(int)  # 类型转换

    return img,label


if __name__=="__main__":
    # dir_name = r'C:\Users\Administrator\Desktop\test\mnist_images_test'
    # data=create_pickle_train(dir_name,28,1) # 图片-->pickle 训练数据
    # print(data.shape)
    # create_pickle_test(dir_name,60,4)  # 图片-->pickle 测试数据
    # img,label=read_and_decode(dir_name+'data.pkl',60,4) # 提起数据和标签
    labels=[1,2,0]
    labels=np.array(labels)
    labels_s=dense_to_one_hot2(labels,3)

    pass