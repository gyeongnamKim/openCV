import cv2
import numpy as np

img = cv2.imread('./data/google_mark.jpg')

(height, width) = img.shape[:2]
center = (width//2,height//2)

mask = np.zeros(img.shape[:2],dtype='uint8')
cv2.circle(mask,center,200,(255,255,255),-1)

masked = cv2.bitwise_and(img,img,mask=mask)

cv2.imshow('mask',masked)

#이미지 저장
cv2.imwrite('./result_image/image_mask.jpg',masked)

cv2.waitKey(0)
cv2.destroyAllWindows()