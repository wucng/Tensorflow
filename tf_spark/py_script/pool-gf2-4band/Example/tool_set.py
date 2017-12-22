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
from collections import Counter


'''
# dir_name 为文件路径
# img_pixel 为图像的像素尺寸
'''


def create_pickle_train(image_path, mask_path, pkl_path, img_pixel=10, channels=3):
    m = 0

    image_data = Multiband2Array(image_path)
    mask_data = gdal.Open(mask_path).ReadAsArray() / 255

    x_size, y_size = mask_data.shape[:2]

    data_list = []

    for i in range(0, x_size - img_pixel + 1, img_pixel // 2):  # 文件夹下的文件名
        if i + img_pixel > x_size:
            i = x_size - img_pixel - 1
        for j in range(0, y_size - img_pixel + 1, img_pixel // 2):
            if j + img_pixel > y_size:
                j = y_size - img_pixel - 1
            cropped_data = image_data[i:i + img_pixel, j:j + img_pixel]
            data1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))  # 展成一行

            # 取频次最多的值作为label
            temp_label = mask_data[i:i + img_pixel, j:j + img_pixel]
            temp_label = temp_label.reshape([img_pixel * img_pixel])
            label_count = dict(Counter(temp_label))
            train_label = max(label_count.items(), key=lambda x: x[1])[0]

            data2 = np.append(data1, train_label)[np.newaxis, :]  # 数据+标签

            data_list.append(data2)

            m += 1

            # if m % 10000 == 0: print(datetime.datetime.now(), "compressed {number} images".format(number=m))
    print(m)
    data_matrix = np.array(data_list, dtype=int)
    del data_list
    data_matrix = data_matrix.reshape((-1, img_pixel * img_pixel * channels + 1))
    with gzip.open(pkl_path, 'wb') as writer:  # 以压缩包方式创建文件，进一步压缩文件
        pickle.dump(data_matrix, writer)  # 数据存储成pickle文件


def read_and_decode(filename):
    with gzip.open(filename, 'rb') as pkl_file:  # 打开文件
        data = pickle.load(pkl_file)  # 加载数据

    return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    # 从标量类标签转换为一个one-hot向量
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    # print index_offset
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

start_index = 0

def next_batch(data, batch_size, img_pixel=3, channels=4):
    global start_index  # 必须定义成全局变量
    global second_index  # 必须定义成全局变量

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

    # 提起出数据和标签
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
    # c4 = src_ds_array[3, :, :]

    # data = cv2.merge([c1, c2, c3, c4])
    data = cv2.merge([c1, c2, c3])

    return data


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("startTime: ", start_time)

    # 生成训练集
    image_path = r"04.tif"
    mask_path = r"04_mask_pool.tif"
    pkl_path = image_path[0:-4] + ".pkl"
    print("影像路径：", image_path)
    print("掩模路径：", mask_path)
    print("序列化文件：", pkl_path)

    create_pickle_train(image_path, mask_path, pkl_path, img_pixel=2, channels=3)

    end_time = datetime.datetime.now()
    print("endTime: ", end_time)
    print("seconds used: ", (end_time - start_time).seconds)
