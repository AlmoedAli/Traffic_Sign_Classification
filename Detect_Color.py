import cv2 as cv
import numpy as  np

cap= cv.VideoCapture(0)
lowerLimit= np.array([136, 87, 111])
upperLimit= np.array([180, 255, 255])
def findBiggestArea(contour):
    c= []
    for i in contour:
        c.append(cv.contourArea(i))
    if len(c)== 0:
        return c, False
    return contour[c.index(max(c))], True

kernel = np.ones((5, 5), "uint8")
while True:
    ret, frame= cap.read()
    image= cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    image= cv.inRange(image, lowerLimit, upperLimit)
    res_red= cv.dilate(image, kernel)
    cv.imshow("kernel", res_red) # is the same with CNN (edge detection using filter or kernel)
    contour, hierarchy= cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    biggestArea, condition= findBiggestArea(contour)
    if condition:
        x, y, w, h= cv.boundingRect(biggestArea)
        Frame= cv.rectangle(frame, (x, y), (x+w, y+ h), (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow("Frame", Frame)
    else:
        cv.imshow("Frame", frame)
    cv.waitKey(10)
    
    