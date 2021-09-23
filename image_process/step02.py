import cv2
import numpy as np

img = cv2.imread('./data/google_mark.jpg')

(height, width) = img.shape[:2]
center = (width//2,height//2)

(b,g,r) = img[0,0]

#이미지 이동
move = np.float32([[1,0,100],[0,1,100]])
moved = cv2.warpAffine(img,move,(width,height))
cv2.imshow("Moved img",moved)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 저장
cv2.imwrite('./result_image/image_move.jpg',moved)

#이미지 회전
move = cv2.getRotationMatrix2D(center,-20,1.0)
rotated = cv2.warpAffine(img,move,(0,0))
cv2.imshow('Rotate',rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 저장
cv2.imwrite('./result_image/image_rotate.jpg',rotated)

#이미지 축소
ratio = 200.0 / width
dimension = (200,int(height * ratio))
resized = cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)
cv2.imshow('resize',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 저장
cv2.imwrite('./result_image/image_resize.jpg',resized)

#좌우대칭
flipped = cv2.flip(img,1)
cv2.imshow("Flipped image",flipped)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 저장
cv2.imwrite('./result_image/image_left_right_flip.jpg',flipped)

#상하대칭
flipped = cv2.flip(img,0)
cv2.imshow("Flipped image",flipped)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 저장
cv2.imwrite('./result_image/image_top_bottom_flip.jpg',flipped)

#상하좌우대칭
flipped = cv2.flip(img,-1)
cv2.imshow("Flipped image",flipped)

#이미지 저장
cv2.imwrite('./result_image/image_all_flip.jpg',flipped)

cv2.waitKey(0)
cv2.destroyAllWindows()
