import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from tkinter import*
image_main = cv2.imread('C:/Users/HP/Desktop/project/images/initnew.jpg')

def filterImage(image):
    img_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_bound = np.array([130,80,175])
    upper_bound = np.array([255,255,255])
    mask = cv2.inRange(img_hsv,lower_bound,upper_bound)
    res = cv2.bitwise_and(image,image,mask=mask)

        
    return res

def drawLines(image,color):
    gray_image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges  = cv2.Canny(gray_image,200,200)

    lines = cv2.HoughLinesP(edges,1,np.pi/180,20,maxLineGap=100)

    for lines in lines:
        x1,y1,x2,y2 = lines[0]
        cv2.line(image,(x1,y1),(x2,y2),color,3)

    return edges

def erodeImage(image,iterations):
    erode = cv2.erode(image,None,iterations=iterations)
    return erode

def dilateImage(image,iterations):
    dilate = cv2.dilate(image,None,iterations=iterations)
    return dilate

def calAngleInit(image):
    b,g,r= cv2.split(image)
    g = np.array(g)
    z = 0
    index=[0,220,0,355]
    z=0
    for x in range(len(b[220])):
        if(b[220][x]== 255):
            index[0]=x
            break
    z=0
    for x in range(len(b[355])):
        if(b[355][x]== 255):
            index[2]=x
            break
    angle = math.degrees(math.atan(abs((index[3]-index[1])/(index[0]-index[2]))))
    return angle

angle=calAngleInit(dilateImage(erodeImage(drawLines(filterImage(image_main),[255,0,0]),3),1))
print('angle : ',angle)
