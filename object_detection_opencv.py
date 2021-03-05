import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

net=cv.dnn.readNet("yolov3-spp.weights","yolov3.cfg")

with open("coco.names","r") as f:
    classes=[line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("loaded")


img=cv.imread('/home/dhananjaya/PycharmProjects/tenserflow/b.jpeg')

height,width,channel=img.shape
blob = cv.dnn.blobFromImage(img, 1.0 / 255.0, (416,416),swapRB=True, crop=False)

net.setInput(blob)
output=net.forward(output_layers)

class_ID=[]
boxes=[]
confidence=[]

for out in output:
    for detection in out:
        score=detection[5:]
        cls_ID=np.argmax(score)
        confi=score[cls_ID]

        if confi>0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x,y,w,h])
            confidence.append(float(confi))
            class_ID.append(cls_ID)

index= cv.dnn.NMSBoxes(boxes, confidence, 0.5, 0.4)
colors = np.random.uniform(0, 255, size=(len(classes), 3))
for i in range(len(boxes)):
    for i in index:
        x, y, w, h = boxes[i]
        label = str(classes[class_ID[i]])
        print("label",label)
        color = colors[class_ID[i]]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(img, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,1 / 2, color, 2)

cv.imshow("image",img)
cv.waitKey(0)

