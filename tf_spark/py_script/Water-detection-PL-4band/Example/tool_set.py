# -*- coding: UTF-8 -*-

"""
针对多于4个波段的影像
使用GDAL读取影像
存储成pickle
"""
import numpy as np
import pickle
import os
import zlib
import gzip
import gdal
import sys
import cv2
import datetime

'''
# dir_name 为文件路径
# img_pixel 为图像的像素尺寸
'''


def create_pickle_train(image_path, mask_path, img_pixel=13, channels=5):
    m = 0

    image_data = Multiband2Array(image_path)
    print("data_matrix_max= ", image_data.max())
    print("data_matrix_min= ", image_data.min())
    mask_data = cv2.split(cv2.imread(mask_path))[0] / 255

    x_size, y_size = image_data.shape[:2]

    data_list = []

    for i in range(0, x_size - img_pixel + 1, img_pixel // 2):  # 文件夹下的文件名
        if i + img_pixel > x_size:
            i = x_size - img_pixel - 1
        for j in range(0, y_size - img_pixel + 1, img_pixel // 2):
            if j + img_pixel > y_size:
                j = y_size - img_pixel - 1
            cropped_data = image_data[i:i + img_pixel, j:j + img_pixel]
            data1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))  # 展成一行
            train_label = mask_data[i+img_pixel//2,j+img_pixel//2]
            data2 = np.append(data1, train_label)[np.newaxis, :]  # 数据+标签

            data_list.append(data2)

            m += 1

            if m % 10000 == 0: print(datetime.datetime.now(), "compressed {number} images".format(number=m))
    print(len(data_list))
    print(m)

    # 注释，将影像压缩成列表，2017.09.25  by xjxf  __start
    data_matrix = np.array(data_list, dtype=int)

    data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * channels+1)))
    # 注释，将影像压缩成列表，2017.09.25  by xjxf  __end


    # with gzip.open('D:/train_data_1-23new.pkl', 'ab') as writer:  # 以压缩包方式创建文件，进一步压缩文件
    #     pickle.dump(data_matrix, writer)  # 数据存储成pickle文件\
    with gzip.open('D:/train_data_64.pkl', 'ab') as writer:  # 以压缩包方式创建文件，进一步压缩文件
        pickle.dump(data_matrix, writer)  # 数据存储成pickle文件


def create_pickle_train_bk(image_path, mask_path, img_pixel=10, channels=3):
    m = 0

    image_data = Multiband2Array(image_path)
    mask_data = cv2.split(cv2.imread(mask_path))[0] / 255

    x_size, y_size = image_data.shape[:2]

    data_matrix = np.zeros([x_size - img_pixel + 1, y_size - img_pixel + 1, img_pixel * img_pixel * channels + 1])

    for i in range(0, x_size - img_pixel + 1, img_pixel):  # 文件夹下的文件名
        if i + img_pixel > x_size:
            i = x_size - img_pixel - 1
        for j in range(0, y_size - img_pixel + 1, img_pixel):
            if j + img_pixel > y_size:
                j = y_size - img_pixel - 1
            cropped_data = image_data[i:i + img_pixel, j:j + img_pixel]
            data1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))  # 展成一行
            train_label = mask_data[i:i + img_pixel, j:j + img_pixel].max()
            data2 = np.append(data1, train_label)[np.newaxis, :]  # 数据+标签

            data_matrix[i, j] = data2

            m += 1

        if m % 10000 == 0: print(datetime.datetime.now(), "compressed {number} images".format(number=m))
    print(i, j)
    train_data = data_matrix.reshape((-1, img_pixel * img_pixel * channels + 1))
    with gzip.open('train_data.pkl', 'wb') as writer:  # 以压缩包方式创建文件，进一步压缩文件
        pickle.dump(train_data, writer)  # 数据存储成pickle文件


# def read_and_decode(filename, img_pixel=isize, channels=img_channel):
def read_and_decode(filename,num_load):
    # data=[]
    num_temp=0
    with gzip.open(filename, 'rb') as pkl_file:  # 打开文件
        for i in range(1, num_load + 1):
            print(i)
            data1 = pickle.load(pkl_file)  # 加载数据
            print("step_1")
            # data1=data2.tolist()

            # # 列表方式压缩解压,2017.09.25,by xjxf  __start
            # num_temp+=len(data1)*(324*2+1)/1024/1024/1024
            # print(num_temp)
            #
            # if i == 1:
            #     data = data1
            # else:
            #     for j in range(len(data1)):
            #         data.append(data1[j])
            # # 列表方式压缩解压,2017.09.25,by xjxf  __end


            # 数组方式压缩解压,2017.09.25,by xjxf  __start
            num_temp += data1.shape[0] * (324 * 2 + 1) / 1024 / 1024 / 1024
            print(num_temp)

            # if 1==i:
            #     data=pickle.load(pkl_file)
            # else:
            #     data=np.append(data,pickle.load(pkl_file),axis=0)
            if i==1:
                #data=data1
                data=np.array(data1)
                print("step_2")
            else :
                #data=np.append(data,data1,axis=0)
                data=np.concatenate((data,data1),axis=0)
                print("step_3")
            # 数组方式压缩解压,2017.09.25,by xjxf  __end


        # data=np.array(data)
    # data=np.array(data)
    # print(data.shape)
   # data=data.reshape((-1,301))
    #data=data1
    return data


'''
其他工具
'''


# ---------------生成多列标签 如：0,1 对应为[1,0],[0,1]------------#
# 单列标签转成多列标签
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    # 从标量类标签转换为一个one-hot向量
    num_labels = labels_dense.shape[0]        #label的行数
    index_offset = np.arange(num_labels) * num_classes
    # print index_offset
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# ------------next_batch------------#
'''
注：
每次 data传入next_batch()完成，进行下一次传入时，先进行打乱
如下面的做法：

total_batch = int(img_nums / batch_size)
data=read_and_decode(filename,img_pixel=isize,channels=3)

for epoch in range(training_epochs):
    # 将数据按行打乱
    index = [i for i in range(len(data))]  # len(data)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    data = data[index]
    for i in range(total_batch):
        img, label=next_batch(data,batch_size,img_pixel=isize,channels=img_channel)
        ......
'''


# 按batch_size提取数据
# batch_size为每次批处理样本数
# data包含特征+标签 每一行都是 特征+标签


def next_batch(data, batch_size, flag, img_pixel=3, channels=4):
    global start_index  # 必须定义成全局变量
    global second_index  # 必须定义成全局变量

    if 1==flag:
        start_index = 0
    # start_index = 0
    second_index = start_index + batch_size

    if second_index > len(data):
        second_index = len(data)

    data1 = data[start_index:second_index]
    # lab=labels[start_index:second_index]
    start_index = second_index
    if start_index >= len(data):
        start_index = 0

    # 将每次得到batch_size个数据按行打乱
    index = [i for i in range(len(data1))]  # len(data1)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    data1 = data1[index]

    # 提取出数据和标签
    img = data1[:, 0:img_pixel * img_pixel * channels]

    # img = img * (1. / img.max) - 0.5
    img = img * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
    img = img.astype(np.float32)  # 类型转换

    label = data1[:, -1]
    label = label.astype(int)  # 类型转换

    return img, label


def Multiband2Array(path):
    src_ds = gdal.Open(path)
    if src_ds is None:
        print('Unable to open %s' % path)
        sys.exit(1)

    src_ds_array = src_ds.ReadAsArray()
    c1 = src_ds_array[0, :, :]
    c2 = src_ds_array[1, :, :]
    c3 = src_ds_array[2, :, :]
    c4 = src_ds_array[3, :, :]

    data = cv2.merge([c1, c2, c3, c4])

    # data = cv2.merge([c1, c2, c3])

    return data


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("startTime: ", start_time)

    # 生成训练集
    for i in range(1, 65):
        image_path = "G:/WaterRecognition/code_cnnnet_64X128/train_data/train_img_" + str(i).zfill(2) + ".tif"
        mask_path = "G:/WaterRecognition/code_cnnnet_64X128/train_data/train_img_mask_" + str(i).zfill(2) + ".tif"
        create_pickle_train(image_path, mask_path, img_pixel=9, channels=4)


    # 生成测试集
    # image_path = "val.tif"
    # mask_path = "val_mask.tif"
    # create_pickle_train(image_path, mask_path, img_pixel=3, channels=3)

    end_time = datetime.datetime.now()
    print("endTime: ", end_time)
    print("seconds used: ", (end_time - start_time).seconds)
