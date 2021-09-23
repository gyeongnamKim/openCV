import cv2
import numpy as np

#YOLO모델 가져오기
#weights 파일 다운로드 필요
net = cv2.dnn.readNet('./data/yolov3.weights','./data/yolov3.cfg')

min_confidence = 0.5
output_name = './result/yolo_result_video.avi'
writer = None

def detectAndDisplay(frame):
    height, width, channels = frame.shape

    #이미지 blob형식으로 변경
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    #박스의 좌표, 신뢰율, 사물 이름 리스트로 저장
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

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #중복되는 박스 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    #이미지에 박스, 사물 이름, confidence 출력
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = '{}: {:.2f}'.format(str(classes[class_ids[i]]),confidences[i]*100)
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5 ), font, 1, color, 1)
            global writer
            if writer is None and output_name is not None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter(output_name, fourcc, 24,
                                         (frame.shape[1], frame.shape[0]), True)
            if writer is not None:
                writer.write(frame)

    cv2.imshow('YOLO video', frame)

classes = []

#분류할 사물 이름 리스트로 저장
with open('./data/coco.names','r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))

#비디오 가져오기
vid = './data/yolo_01.mp4'

#비디오 캡쳐
cap = cv2.VideoCapture(vid)
if not cap.isOpened:
    print('opening video capture error')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('no captured frame')
        break
    detectAndDisplay(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()