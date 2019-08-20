import numpy as np
import cv2
import os

cap = cv2.VideoCapture('sample.3gp')

def drawLine(image):
    image= cv2.resize(image,(720,512))
    img_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([130,80,175])
    upper_blue = np.array([255,255,255])
    mask = cv2.inRange(img_hsv,lower_blue,upper_blue)
    res = cv2.bitwise_and(image,image,mask=mask)

    gray= cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    edges  = cv2.Canny(gray,75,100)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,20,maxLineGap=500)

    for lines in lines:
        x1,y1,x2,y2 = lines[0]
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)

    return image,res

def erodeImage(image):
    kernal=np.ones((3,3),np.uint8)
    struct = cv2.getStructuringElement(cv2.MORPH_ERODE,(3,3))
    image=cv2.erode(image,struct,iterations=1)
    return image

num =1;
while(1):
    # take each frame
    ret,frame = cap.read()

    image= cv2.resize(frame,(720,512))
    if num==1:
        cv2.imwrite(os.path.join('','init.png'),image)
    
    cv2.imshow('frame',image)

    #convert BGR to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([128,78,169])
    upper_blue = np.array([200,200,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    res = erodeImage(res)
    image,res = drawLine(res)

    cv2.imshow('mask',image)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
