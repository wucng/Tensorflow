# -*- coding: UTF-8 -*-

import cv2
import numpy as np

# Load image and template
img = cv2.imread("E:/07/L15-3345E-2311N.tif", 0)
# des=cv2.bitwise_not(img)
_,des=cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

# cv2.normalize(des,des,0,1,cv2.NORM_MINMAX,-1)
des=cv2.bitwise_not(des)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
res = cv2.morphologyEx(des,cv2.MORPH_OPEN,kernel,iterations=5)

cv2.namedWindow("dst",cv2.WINDOW_FREERATIO)
cv2.imshow("dst",res)
cv2.waitKey(0)
cv2.destroyAllWindows()
