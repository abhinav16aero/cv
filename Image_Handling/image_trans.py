import cv2 as cv
import numpy as np
img = cv.imread('C:/Users/Anupriya/Desktop/Edity/Img1.png')
cv.imshow('me', img)

blank = np.zeros(img.shape, dtype='uint8')
#cv.imshow('Blank', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
#cv.imshow('Blur', blur)

ret, thresh = cv.threshold(gray, 115, 100, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

canny = cv.Canny(gray, 135, 115)
cv.imshow('Canny Edges', canny)
#cv.drawContours(blank, contours, -1, (0,0,255), 1)

#cv.imshow('Contours Drawn', canny)

#print(f'{len(contours)} contour(s) found!')

cv.waitKey(0)