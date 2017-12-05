# -*- coding: utf-8 -*-

"""
云检测及估计云量
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread("E:/07/L15-3345E-2312N.tif")
# img=cv2.resize(img,(480,480))

# img=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
# img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# H,S,V=cv2.split(img)

# hsv=[170,40,215]
# thresh = 40

hsv=[120,80,190]
thresh = 80

bgr=cv2.cvtColor(np.uint8([[hsv]]),cv2.COLOR_HSV2BGR)[0][0]
print(bgr)

minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

maskHSV = cv2.inRange(img, minHSV, maxHSV)

resultHSV = cv2.bitwise_and(img, img, mask = maskHSV)
H,S,V=cv2.split(resultHSV)


# V=cv2.medianBlur(V,5) # 滤波
# kernel = np.ones((5, 5), np.uint8)
# V = cv2.dilate(V, kernel, iterations=1) # 膨胀
# kernel = np.ones((5, 5), np.uint8)
# V = cv2.erode(V, kernel, iterations=1) # 腐蚀
# V=cv2.medianBlur(V,5) # 滤波

V=np.ceil(V/255)
print("云量：",np.sum(V)/(V.shape[0]*V.shape[1]))

plt.subplot(2,2,1);plt.imshow(resultHSV,"gray");plt.title("resultHSV")
plt.subplot(2,2,2);plt.imshow(H,"gray");plt.title("H")
plt.subplot(2,2,3);plt.imshow(S,"gray");plt.title("S")
plt.subplot(2,2,4);plt.imshow(V,"gray");plt.title("V")
plt.show()
