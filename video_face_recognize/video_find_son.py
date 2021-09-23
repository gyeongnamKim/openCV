import cv2
import face_recognition
import pickle
import time

file_name = './data/video/son_02.mp4'
encoding_file = './data/encodings.pickle'
unknown_name = 'Unknown'
model_method = 'hog'
output_name = './result/result_son_video.avi'

#캡쳐 이미지 박스, 텍스트 추가
def detectAndDisplay(image):
    start_time = time.time()
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model= model_method)
    encodings_1 = face_recognition.face_encodings(rgb,boxes)
    names = []
    for encoding in encodings_1:
        matches = face_recognition.compare_faces(data['encodings'],encoding)
        name = unknown_name
        if True in matches:
            matchedIdxs = [i for (i,b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data['names'][i]
                counts[name] = counts.get(name,0)+1
            name = max(counts,key=counts.get)
        names.append(name)
    for ((top,right,bottom,left),name) in zip(boxes,names):
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0,255,0)
        line = 2
        if (name == unknown_name):
            color = (0,0,255)
            line = 1
            name = ''
        cv2.rectangle(image,(left,top),(right,bottom),color,line)
        cv2.putText(image,name,(left,y),cv2.FONT_HERSHEY_COMPLEX,0.75,color,line)
    end_time = time.time()
    process_time = int(end_time - start_time)
    print('Run time',process_time)
    image = cv2.resize(image,None,fx=0.5,fy=0.5)
    cv2.imshow('Recognition',image)

    #비디오 저장
    global writer
    if writer is None and output_name is not None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(output_name,fourcc,24,
                                 (image.shape[1],image.shape[0]),True)
    if writer is not None:
        writer.write(image)
data = pickle.loads(open(encoding_file,'rb').read())

#비디오 캡쳐 프레임 이미지화
cap = cv2.VideoCapture(file_name)
writer = None
if not cap.isOpened:
    print('Opening video capture ERROR')
    exit(0)
while True:
    ret, frame = cap.read()
    #캡처가 끝나면 프로그램 종료
    if frame is None:
        print('No capture frame ERROR')
        cap.release()
        writer.release()
        break
    detectAndDisplay(frame)
    #q를 입력 받으면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()