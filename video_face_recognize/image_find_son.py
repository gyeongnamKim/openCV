import cv2
import face_recognition
import pickle
import time

image_file = './data/image/soccer_01.jpg'
encoding_file = './data/encodings.pickle'
unknown_name = 'Unknown'
model_method = 'cnn'


def detectAndDisplay(frame):
    start_time = time.time()
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb,model=model_method)
    encodings_1 = face_recognition.face_encodings(rgb,boxes)
    names = []
    for encoding in encodings_1:
        maches = face_recognition.compare_faces(data['encodings'],encoding)
        name = unknown_name
        if True in maches:
            machedindexs = [i for (i,b) in enumerate(maches) if b]
            counts = {}
            for i in machedindexs:
                name = data['names'][i]
                counts[name] = counts.get(name,0)+1
            name = max(counts,key=counts.get)
        names.append(name)
    for ((top,right,bottom,left),name) in zip(boxes,names):
        y= top -15 if top -15 >15 else top +15
        color = (0,255,0)
        line = 2
        if (name == unknown_name):
            color = (0,0,255)
            line = 1
            name = ''
        cv2.rectangle(frame,(left,top),(right,bottom),color,line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame,name,(left,y),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,line)

    end_time = time.time()
    process_time = end_time - start_time
    print('process time',process_time)
    cv2.imshow('Recognition',frame)
    cv2.imwrite('./result/result_soccer_1.jpg',frame)


data = pickle.loads(open(encoding_file,'rb').read())

image = cv2.imread(image_file)
detectAndDisplay(image)

cv2.waitKey(0)
cv2.destroyAllWindows()