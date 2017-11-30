#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
解析本地mnist图片的路径，与gtdata路径结合拼成gtdata的完整路径
结果 如：
0/0_0.bmp
0/0_1.bmp
0/0_10.bmp
……
8/8_189.bmp
8/8_19.bmp
"""

import glob
import os

img_paths=glob.glob(r"D:\mnist_images\*\*")

fp=open('mnist_name.txt','w')

for img_path in img_paths:
    # fp.write(img_path.split('\\')[-1]) # 写入文件名

    # fp.write(os.path.join(img_path.split('\\')[-2],img_path.split('\\')[-1])+'\n')
    fp.write(img_path.split('\\')[-2]+'/'+ img_path.split('\\')[-1] + '\n')

fp.close()
