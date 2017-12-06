# -*- coding: UTF-8 -*-
"""
读取影像与对应掩膜图像，生成影像数据与对应的标签
裁剪影像大小为9x9x4 
"""
import numpy as np
import pickle
import os
import zlib
import gzip
import gdal
import sys
# import cv2
import datetime


def create_pickle_train(image_path, mask_path, img_pixel=9, channels=4):
    m = 0

    image_data = Multiband2Array(image_path)
    print("data_matrix_max= ", image_data.max())
    print("data_matrix_min= ", image_data.min())
    # mask_data = cv2.split(cv2.imread(mask_path))[0] / 255
    mask_data=Multiband2Array(mask_path)/255

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
            train_label = mask_data[i+img_pixel//2,j+img_pixel//2] # 取 9x9 中间位置的掩膜像素值做标签
            data2 = np.append(data1, train_label)[np.newaxis, :]  # 数据+标签

            data_list.append(data2)

            m += 1

            if m % 10000 == 0: print(datetime.datetime.now(), "compressed {number} images".format(number=m))
    print(len(data_list))
    print(m)

    data_matrix = np.array(data_list, dtype=int)

    data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * channels+1)))

    """
    with gzip.open('D:/train_data_64.pkl', 'ab') as writer:  # 以压缩包方式创建文件，进一步压缩文件
        pickle.dump(data_matrix, writer)  # 数据存储成pickle文件
    """
    return data_matrix # shape [none,9*9*4+1]


def Multiband2Array(path):

    src_ds = gdal.Open(path)
    if src_ds is None:
        print('Unable to open %s'% path)
        sys.exit(1)

    xcount=src_ds.RasterXSize # 宽度
    ycount=src_ds.RasterYSize # 高度
    ibands=src_ds.RasterCount # 波段数

    # print "[ RASTER BAND COUNT ]: ", ibands
    # if ibands==4:ibands=3
    for band in range(ibands):
        band += 1
        # print "[ GETTING BAND ]: ", band
        srcband = src_ds.GetRasterBand(band) # 获取该波段
        if srcband is None:
            continue

        # Read raster as arrays 类似RasterIO（C++）
        dataraster = srcband.ReadAsArray(0, 0, xcount, ycount).astype(np.float16)
        if band==1:
            data=dataraster.reshape((ycount,xcount,1))
        else:
            # 将每个波段的数组很并到一个3维数组中
            data=np.append(data,dataraster.reshape((ycount,xcount,1)),axis=2)

    return data


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print("startTime: ", start_time)

    # 生成训练集
    # for i in range(1, 65):
    #     image_path = "G:/WaterRecognition/code_cnnnet_64X128/train_data/train_img_" + str(i).zfill(2) + ".tif"
    #     mask_path = "G:/WaterRecognition/code_cnnnet_64X128/train_data/train_img_mask_" + str(i).zfill(2) + ".tif"
    #     create_pickle_train(image_path, mask_path, img_pixel=9, channels=4)

    image_path = r"E:\项目\AI-Waterdetection-PL-4band\train_data\train_img_01.tif"
    mask_path = r"E:\项目\AI-Waterdetection-PL-4band\train_data\train_img_mask_01.tif"
    create_pickle_train(image_path, mask_path, img_pixel=9, channels=4)

    # 生成测试集
    # image_path = "val.tif"
    # mask_path = "val_mask.tif"
    # create_pickle_train(image_path, mask_path, img_pixel=3, channels=3)

    end_time = datetime.datetime.now()
    print("endTime: ", end_time)
    print("seconds used: ", (end_time - start_time).seconds)
