import cv2 as cv
import numpy as np

img = cv.imread('C:/Users/Anupriya/Desktop/Edity/Img1.png')
cv.imshow('Cs', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('adaptive', adaptive)

cv.waitKey(0)