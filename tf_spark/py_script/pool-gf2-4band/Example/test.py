#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function
from os import path
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

import numpy as np
import datetime
from osgeo import gdal
import cv2
import sys


starttime = datetime.datetime.now()
print(starttime)
# 训练集文件路径
step = 2
dir_name = 'train_img/04.tif'
model_path = "model/"


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

    return data


def return_list(image_path, img_pixel=9, channels=4):
    m = 0
    image_data = Multiband2Array(image_path)
    x_size, y_size = image_data.shape[:2]
    data_list = []
    for i in range(0, x_size - img_pixel + 1):
        if i + img_pixel > x_size:
            i = x_size - img_pixel - 1
        for j in range(0, y_size - img_pixel + 1):
            if j + img_pixel > y_size:
                j = y_size - img_pixel - 1
            cropped_data = image_data[i:i + img_pixel, j:j + img_pixel]
            data1 = cropped_data.reshape((-1, img_pixel * img_pixel * channels))
            data_list.append(data1)
            m += 1
            # if m % 1000000 == 0:
            #     print(datetime.datetime.now(), "compressed {number} images".format(number=m))
    print(m)
    return i+1, j+1, data_list

# time格式化
print("startTime: ", starttime)

'''
# CNN 完整程序  测试模型
'''
training_epochs = 50
batch_size = 1000
isize = 2
channels = 4

# Network Parameters
img_size = isize * isize * channels  # data input (img shape: 28*28*3)
label_cols = 2  # total classes (云、非云)使用标签[1,0,0] 3维

if __name__ == '__main__':

    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")

    srcDS = gdal.Open(dir_name, gdal.GA_ReadOnly)  # 只读方式打开原始影像

    geoTrans = srcDS.GetGeoTransform()  # 获取地理参考6参数
    srcPro = srcDS.GetProjection()  # 获取坐标引用
    srcXSize = srcDS.RasterXSize  # 宽度
    srcYSize = srcDS.RasterYSize  # 高度
    nbands = srcDS.RasterCount  # 波段数
    # print("nbands = :", nbands)

    h_size, w_size, date_list = return_list(dir_name, img_pixel=isize, channels=channels)
    # print(h_size, w_size, len(date_list))

    shortname, extension = path.splitext(dir_name)
    raster_fn = shortname + "_mask_" + "_step" + str(step).zfill(2) + ".tiff"

    with tf.Session() as sess:
        # 定义输入、输出
        signature_key = 'predict_images'
        input_key = 'images'
        output_key = 'scores'
        keep_prob_key = "keep_prob"

        meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

        # 从meta_graph_def中取出SignatureDef对象
        signature = meta_graph_def.signature_def

        # 从signature中找出具体输入输出的tensor name
        x_tensor_name = signature[signature_key].inputs[input_key].name
        y_tensor_name = signature[signature_key].outputs[output_key].name
        keep_prob_tensor_name = signature[signature_key].inputs[keep_prob_key].name

        # 获取tensor 并inference
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)
        keep_prob = sess.graph.get_tensor_by_name(keep_prob_tensor_name)

        result = []
        n = 0
        for i in range(0, len(date_list), batch_size):
            if i + batch_size > len(date_list):
                batch_size = len(date_list) - i
            cropped_data = (np.array(date_list[i:i+batch_size]) / 255.0) - 0.5
            cropped_data = cropped_data.reshape([-1, img_size])
            pred1 = sess.run(y, feed_dict={x: cropped_data, keep_prob: 1.})
            result.extend(pred1[:, 1])
        result = np.array(result)
        img = (np.reshape(result, [int(len(date_list)/w_size), w_size])*255).astype(np.uint8)
        cv2.imwrite(raster_fn, img)

        # 添加坐标
        target_ds = gdal.Open(raster_fn, gdal.GA_Update)
        target_ds.SetGeoTransform(geoTrans)
        target_ds.SetProjection(srcPro)
        target_ds.FlushCache()

        endtime = datetime.datetime.now()
        print("endtime: ", endtime)
        print("time used in seconds: ", (endtime - starttime).seconds)
