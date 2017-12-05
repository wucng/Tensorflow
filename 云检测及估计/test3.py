# -*- coding: utf-8 -*-

"""
云检测及云量估计
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread("D:/cloud.tif")
# img=cv2.resize(img,(480,480))

# img=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
H,S,V=cv2.split(img)

# mask=(V>235).astype(np.uint8)*255
# 或
mask=(V>235).astype(np.uint8)
mask = cv2.bitwise_and(V, V, mask = mask)

mask=cv2.medianBlur(mask,5) # 滤波

# 估计云量
cloud_cover=np.sum(np.ceil(mask/255))/(mask.shape[0]*mask.shape[1])
print("云量：%.3f%%" % (cloud_cover*100))

# plt.subplot(2,2,1);plt.imshow(img,"gray");plt.title("resultHSV")
# plt.subplot(2,2,2);plt.imshow(H,"gray");plt.title("H")
# plt.subplot(2,2,3);plt.imshow(S,"gray");plt.title("S")
# plt.subplot(2,2,4);plt.imshow(V,"gray");plt.title("V")

plt.figure();plt.imshow(mask,"gray")
plt.show()
