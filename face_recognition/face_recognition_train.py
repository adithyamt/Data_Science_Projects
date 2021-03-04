import os
import cv2 as cv
import numpy as np

data='/home/dhananjaya/adi/data/train'
stars=[]

harr_cascade=cv.CascadeClassifier('haar_face.xml')

for i in os.listdir('/home/dhananjaya/adi/data/train'):
    stars.append(i)
print(stars)

feature=[]
labels=[]

def create_train():
    for celeb in stars:
        path=os.path.join(data,celeb)
        label=stars.index(celeb)

        for img in os.listdir(path):
            img_array=cv.imread(os.path.join(path,img))
            gray=cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            harr_cascade = cv.CascadeClassifier('haar_face.xml')

            face_rect=harr_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for x,y,w,h in face_rect:
                face_roi=gray[y:y+h,x:x+w]
                feature.append(face_roi)
                labels.append(label)
create_train()

print("length of the features {}".format(len(feature)))
print("length of the labels {}".format(len(labels)))

feature=np.array(feature,dtype='object')
labels=np.array(labels)

face_recognizer=cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(feature,labels)

np.save('feature.npy',feature)
np.save('labels.npy',labels)