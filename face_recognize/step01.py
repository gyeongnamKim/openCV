#pip install cmake
#pip install dlib
#pip install face_recognotion
import pickle
import cv2
import face_recognition

datasets_paths = ['./data/son/','./data/tedy/']
names = ['Son','Tedy']
number_images = 10
image_type = '.jpg'
encoding_file = './data/encodings.pickle'
model_method = 'cnn'

knownEncodings = []
knownNmaes = []

for (i,datasets_path) in enumerate(datasets_paths):
    name = names[i]
    for idx in range(number_images):
        file_name = datasets_path +str(idx+1) + image_type
        image = cv2.imread(file_name)
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb,model = model_method)

        encodings_1 = face_recognition.face_encodings(rgb,boxes)
        for encoding in encodings_1:
            print(file_name,name,encoding)
            knownEncodings.append(encoding)
            knownNmaes.append(name)

data = {'encodings':knownEncodings,'names':knownNmaes}
f = open(encoding_file,'wb')
f.write(pickle.dumps(data))
f.close()