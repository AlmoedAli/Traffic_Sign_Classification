import cv2 as cv
from keras.models import load_model
import numpy as np

classes= {0: "NO CAR", 1: "STOP", 2: "NOT TURN LEFT", 3: "NO WALKING", 4: "SLOWING", 5: "NO REVERSING", 6: "Invalidation"}

def returnRedness(img):
    yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
    y, u, v = cv.split(yuv)
    return v

def threshold(img,T= 160):
	_, img = cv.threshold(img,T,255,cv.THRESH_BINARY)
	return img 

def findContour(img):
	contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
	return contours

def findBiggestAreaContour(contours):
    c = [cv.contourArea(i) for i in contours]
    if len(c) == 0:
        return c, False
    else:
        return contours[c.index(max(c))], True

def DrawBoxes(img,contours):
	x, y, w, h = cv.boundingRect(contours)
	img = cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
	sign = img[y:(y+h) , x:(x+w)]
	return img, sign, x, y
    
def Implement(model):
    cap= cv.VideoCapture(0)
    while True:
        _, frame = cap.read()
        redness = returnRedness(frame)
        thresh = threshold(redness) 	
        contours = findContour(thresh)
        big, ret = findBiggestAreaContour(contours)
        if ret:
            if cv.contourArea(big) > 4000:
                img, sign, x, y = DrawBoxes(frame,big)
                sign = cv.flip(sign, 1)
                # cv.imshow('Frame',img)
                image= cv.resize(sign, (64, 64))
                Data= np.array(image).reshape((1, 64, 64, 3))
                label= classes[np.argmax(model.predict(Data))]
                img= cv.putText(img, str(label),  (x, y), 2, 1, (255, 0, 0), 3)
                cv.imshow('Frame',img)
            else:
                cv.imshow("Frame", frame)
        else:
            cv.imshow("Frame", frame)
        cv.waitKey(10)
            
if __name__== "__main__":
    model= load_model('mode.h5')
    Implement(model)    
    

