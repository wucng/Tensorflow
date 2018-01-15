#! /usr/bin/python
# -*- coding: utf8 -*-

import os
import glob
import shutil

img_paths=[]
[img_paths.append(i.replace('\n', ''))
     for i in open('./tif.log')]

imgs=glob.glob('/home/wucong/*.tif')+glob.glob('/home/wucong/image_walter/*.tif')
print(len(imgs))

img_16=img_paths[7434:11151]
img_22=img_paths[29736:33453]

path_16='/home/wucong/16'
path_22='/home/wucong/22'
if not os.path.exists(path_16):os.mkdir(path_16)
if not os.path.exists(path_22):os.mkdir(path_22)


for img in imgs:
    img_1=img.split('/')[-1].split('_')[0]+'.tif'
    try:
        if img_1 in img_16:shutil.move(img,path_16)
        if img_1 in img_22:shutil.move(img,path_22)
    except:
        continue
    # exit(-1)
