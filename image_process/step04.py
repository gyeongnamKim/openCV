import cv2
import numpy as np

img = cv2.imread('./data/google_mark.jpg')

(Blue,Green,Red) = cv2.split(img)

cv2.imshow('Red',Red)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Green',Green)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Blue',Blue)
cv2.waitKey(0)
cv2.destroyAllWindows()

zeros = np.zeros(img.shape[:2],dtype='uint8')
cv2.imshow('Red',cv2.merge([zeros,zeros,Red]))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Green',cv2.merge([zeros,Green,zeros]))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Blue',cv2.merge([Blue,zeros,zeros]))
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
cv2.imshow('lab',lab)

cv2.waitKey(0)
cv2.destroyAllWindows()
