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
bid_dict={
    0:1,
    15:1,
    16:2,
    17:3,
    21:4,
    30:5,
    41:6,
    43:7,
    46:8,
    50:9,
    54:10,
    56:11,
    65:12,
    90:13
}


global num_img

def create_pickle_train(image_path, mask_path, filename,img_pixel=400, channels=4):
    m = 0
    compress_count=0  #增加一个值，来返回压缩了几次  by bxjxf
    ++num_img
    # gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # mask_img= gdal.Open(mask_path, gdal.GA_ReadOnly)  # 只读方式打开原始影像
    # srcXSize = mask_img.RasterXSize  # 宽度
    # srcYSize = mask_img.RasterYSize  # 高度
    # band=mask_img.GetRasterBand(1)
    # mask_data=band.ReadAsArray(0,0,srcYSize,srcXSize)
    step=img_pixel//3
    mask_img=gdal.Open(mask_path)
    mask_data_temp=mask_img.ReadAsArray()
    # print(mask_data.shape)
    row_num,col_num=mask_data_temp.shape
    mask_data=np.zeros([row_num,col_num])
    mask_data=mask_data_temp
    # mask_data=mask_data_temp
    # #将图像中的编码映射成序列数   2017.10.16,by xjxf  __start
    # for i_1 in range(row_num):
    #     for j_1 in range(col_num):
    #         if mask_data_temp[i_1,j_1] ==2 or mask_data_temp[i_1,j_1]==50:
    #             mask_data[i_1,j_1]=1
    #         elif mask_data_temp[i_1,j_1]==3 or mask_data_temp[i_1,j_1]==54:
    #             mask_data[i_1,j_1]=2
    #         else:
    #             mask_data[i_1, j_1] = 0
    # mask_data_new=mask_data.reshape([row_num,col_num])
    # cv2.imwrite("xjxf.tif",mask_data_new)
    # print(num_img)
    # # 将图像中的编码映射成序列数   2017.10.16,by xjxf  __end

    image_data = Multiband2Array(image_path)

    # mask_data = cv2.split(cv2.imread(mask_path))[0]

    x_size, y_size = image_data.shape[:2]

    data_list = []
    flag_x=True
    flag_y=True
    # print(len(data_list))

    for i in range(0, x_size - img_pixel + 1, step):  # 文件夹下的文件名
        i_end=i+img_pixel
        if i + img_pixel > x_size:
            # i = x_size - img_pixel - 1
            i_end=x_size
            flag_x=False

        flag_y=False
        for j in range(0, y_size - img_pixel + 1,step):
            j_end=j+img_pixel
            if j + img_pixel > y_size:
            #     j = y_size - img_pixel - 1
                j_end=y_size
                flag_y=False

            cropped_data_temp = image_data[i:i_end, j:j_end]
            #对截取的样本做扩充, 2017.10.24, by xjxf __start

            cropped_data=np.lib.pad(cropped_data_temp,((0,img_pixel-(i_end-i)),(0,img_pixel-(j_end-j)),(0,0)),'constant',constant_values=0)
            # 对截取的样本做扩充, 2017.10.24, by xjxf __end


            data_1 = cropped_data.reshape((-1, img_pixel * img_pixel*channels ))  # 展成一行
            # cropped_data_2 = image_data[i:i + img_pixel, j:j + img_pixel, 1]
            # data_2 = cropped_data_2.reshape((-1, img_pixel * img_pixel ))  # 展成一行
            # cropped_data_3 = image_data[i:i + img_pixel, j:j + img_pixel, 2]
            # data_3 = cropped_data_3.reshape((-1, img_pixel * img_pixel ))  # 展成一行
            cropped_mask_data_temp=mask_data[i:i_end,j:j_end]
            # 对截取的样本做扩充, 2017.10.24, by xjxf __start
            cropped_mask_data=np.lib.pad(cropped_mask_data_temp,((0,img_pixel-(i_end-i)),(0,img_pixel-(j_end-j))),'constant',constant_values=0)
            # 对截取的样本做扩充, 2017.10.24, by xjxf __end
            train_label = cropped_mask_data.reshape((-1,img_pixel*img_pixel))

            # data2 = np.append(data_1[np.newaxis,:], data_2[np.newaxis,:])
            # data2=np.append(data2,data_3[np.newaxis,:])

            data2=np.append(data_1,train_label)[np.newaxis,:]


            # if train_label==0 or train_label==1 or train_label==3 or train_label==5:    #去除标签是其他的样本
            # if train_label==1:    #去除标签是其他的样本
            #     print("hello")
            data_list.append(data2)
            m += 1

            # if m % 10000 == 0: print(datetime.datetime.now(), "compressed {number} images".format(number=m))

            # 每到一百万张采样图片时，保存压缩一次到硬盘  ___start_by xjxf
            if m%1000000==0:
                compress_count += 1
                print("第"+str(compress_count)+"次压缩")
                data_matrix = np.array(data_list, dtype=int)
                data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * (channels + 1))))
                with gzip.open(filename, 'ab') as writer:  # 以压缩包方式创建文件，进一步压缩文件
                    pickle.dump(data_matrix, writer)  # 数据存储成pickle文件
                data_list=[]

            # 每到一万张采样图片时，保存压缩一次到硬盘  ___end_by xjxf


    print(m)
    #将最后一部分取样也做压缩，__start,by xjxf
    if len(data_list)> 0:
        data_matrix = np.array(data_list, dtype=int)
        data_matrix = data_matrix.reshape((-1, (img_pixel * img_pixel * (channels + 1))))
        with gzip.open(filename,'ab') as writer:  # 以压缩包方式创建文件，进一步压缩文件
            pickle.dump(data_matrix, writer)  # 数据存储成pickle文件
        compress_count+=1
    return compress_count
    # 将最后一部分取样也做压缩，__end,by xjxf



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
def read_and_decode(filename,number):
    # data=[]
    with gzip.open(filename, 'rb') as pkl_file:  # 打开文件
        for i in range(1,number+1):

            data1 = pickle.load(pkl_file)  # 加载数据
            if i==1:
                data=data1
                data=np.array(data)
            else :
                data=np.append(data,data1,axis=0)
            # print(i)
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


def next_batch(data, batch_size,flag, img_pixel=3, channels=4):
    global start_index  # 必须定义成全局变量
    global second_index  # 必须定义成全局变量

    if 1 == flag:
        start_index = 0


    second_index = start_index + batch_size

    if second_index > len(data):
        second_index = len(data)
    # print("start_index:"+str(start_index)+"end_index:"+str(second_index))
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
    img = data1[:, 0:img_pixel*img_pixel*channels]

    # img = img * (1. / img.max) - 0.5
    img = img * (1. / 255) - 0.5  # 数据归一化到 -0.5～0.5
    img = img.astype(np.float32)  # 类型转换

    label = data1[:,img_pixel*img_pixel*channels:img_pixel*img_pixel*(channels+1)]
    label=label.reshape([-1,1])
    # label_mask=label.reshape([img_pixel,img_pixel])*255
    # cv2.imwrite("mask_review.tif",label_mask)
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

    compress_count=0

    # 生成训练集
    num_img=0

    for i in range(19,22):
        # image_path = "PL_img/train_img_" + str(i).zfill(2) + ".tif"
        # mask_path = "PL_img/train_img_mask_" + str(i).zfill(2) + ".tif"
        file_compress_count = open("压缩次数.txt", "w+")


        # if i is not 14:
        start_time_1 = datetime.datetime.now()
        print("picture",str(i).zfill(2),":started at:",start_time_1)


        image_path = "train_data/" + str(i).zfill(2) + ".tif"
        mask_path = "train_data/"  +str(i).zfill(2)+  "_mask"+ ".tif"
        filename = "train_data_400_all" + ""+ ".pkl"
        compress_count=create_pickle_train(image_path, mask_path,filename,img_pixel=400, channels=3)
        end_time_1 = datetime.datetime.now()
        print("picture", str(i).zfill(2), ":finished at:", end_time_1)
        # create_pickle_train(image_path, mask_path, filename, img_pixel=13, channels=3)
        # compress_count+=1
        file_compress_count.write(str(compress_count)+"\n")
        # file_compress_count.close()


    end_time = datetime.datetime.now()
    print("endTime: ", end_time)
    print("seconds used: ", (end_time - start_time).seconds)
