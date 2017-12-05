# -*- coding: utf-8 -*-

"""
云检测及估计云量
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img=cv2.imread("D:/cloud.tif")

gr=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
bg = gr.copy()
for r in range(1,5):
    kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
    bg=cv2.morphologyEx(bg,cv2.MORPH_OPEN,kernel2)
    bg=cv2.morphologyEx(bg,cv2.MORPH_CLOSE,kernel2)

dif = bg - gr

_,bw=cv2.threshold(dif,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
_,dark=cv2.threshold(bg,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

darkpix=[]
for r in range(dark.shape[0]):
    for c in range(dark.shape[1]):
        if not dark[r,c]: # 找到白色区域
            darkpix.append(gr[r,c])

_,darkpix=cv2.threshold(np.array(darkpix),0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)

index=0
for r in range(dark.shape[0]):
    for c in range(dark.shape[1]):
        if not dark[r,c]: # 找到白色区域
            bw[r,c]=darkpix[index]
            index+=1

res=cv2.bitwise_xor(gr,bw)

dst=(res<16).astype(np.uint8)*255

for r in range(1,5):
    kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
    dst=cv2.morphologyEx(dst,cv2.MORPH_OPEN,kernel2)
    dst=cv2.morphologyEx(dst,cv2.MORPH_CLOSE,kernel2)

dst=cv2.medianBlur(dst,5)

cv2.namedWindow("bg",cv2.WINDOW_FREERATIO)
cv2.imshow("bg",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
