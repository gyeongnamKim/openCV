import cv2
import numpy as np

#YOLO모델 가져오기
#weights 파일 다운로드 필요
net = cv2.dnn.readNet('./data/yolov3.weights','./data/yolov3.cfg')



min_confidence = 0.5
classes = []

#분류할 사물 이름 파일 읽어와서 리스트로 저장
with open('./data/coco.names','r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))

#이미지 가져오기
#이미지 출처 http://www.kyeonggi.com/news/galleryView.html?idxno=1494729
img = cv2.imread('./data/yolo_test_image.jpg')
height, width, channels = img.shape

#blob 형식으로 변경 (320,320)은 빠르고 (609,609)는 정확도가 올라간다.
#중간인 (416,416)사이즈로 지정
blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)

#모델에 로드
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

#빈 리스트에 사각형(박스)과 사물 이름 추가
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > min_confidence:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
#중복되는 박스 제거
indexes = cv2.dnn.NMSBoxes(boxes,confidences,min_confidence,0.4)

font = cv2.FONT_HERSHEY_PLAIN

#리스트를 이용하여 이미지에 박스와 사물 이름 추가
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,label,(x,y+30),font,2,color,2)

#이미지 출력
img = cv2.resize(img,None,fx=2,fy=2)
cv2.imshow('YOLO image',img)

cv2.imwrite('./result/yolo_result_image.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()