import cv2
print(cv2.__version__)

img = cv2.imread('./data/google_mark.jpg')
print(f'width:{img.shape[1]}')
print(f'height:{img.shape[0]}')
print(f'channels:{img.shape[2]}')

cv2.imshow('google logo',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#openCV는 RGB를 반대로 입력
(b,g,r) = img[0,0]
print(f'Pixel at (0,0) - Red:{r},Green{g},Blue:{b}')

dot = img[50:100,50:100]
cv2.imshow("Dot",dot)
cv2.waitKey(0)
cv2.destroyAllWindows()

img[50:100,50:100] = (0,0,0)

#사각형 그려주는 함수
cv2.rectangle(img,(150,50),(200,100),(0,255,0),5)

#원 그려주는 함수
cv2.circle(img,(275,75),25,(0,255,255),-1)

#라인 그리기
cv2.line(img,(350,100),(400,100),(255,0,0),5)

#텍스트 작성
cv2.putText(img,'google',(0,400),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),4)
cv2.imshow('google-drow',img)

#최종 이미지 저장
cv2.imwrite('./result_image/image_draw.jpg',img)

#키를 입력하면 이미지 창을 닫아주는 함수
cv2.waitKey(0)
cv2.destroyAllWindows()




