import cv2 as cv
import os
import numpy as np
img=cv.imread('/home/dhananjaya/Downloads/dhoni.jpg')
cv.imshow('dhoni',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

harr_cascade=cv.CascadeClassifier('haar_face.xml')
face_rect=harr_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)

# need to change minNeighbour value to get accurate face detection
print('number of faces found={}'.format(len(face_rect)))
for(x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.putText(img,"face",(x,y),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),thickness=1)
cv.imshow('detected_face',img)

cv.waitKey(0)

