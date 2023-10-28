import cv2 as cv
import numpy as np

img = cv.imread('C:/Users/Anupriya/Desktop/Edity/Img1.png')
cv.imshow('Cs', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank Image', blank)

circle = cv.circle(blank.copy(), (img.shape[1]//2 +100 ,img.shape[0]//2 +70), 400, 525, -1)

rectangle = cv.rectangle(blank.copy(), (100,100), (650,650), 255, -1)

weird_shape = cv.bitwise_and(circle,rectangle)
cv.imshow('Weird Shape', weird_shape)

masked = cv.bitwise_and(img,img,mask=weird_shape)
cv.imshow('Weird Shaped Masked Image', masked)

cv.waitKey(0)