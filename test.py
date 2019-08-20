import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

image_main = cv2.imread('C:/Users/S.S.Aludeniya/Desktop/project/images/initnew.jpg')
image_read = cv2.imread('C:/Users/S.S.Aludeniya/Desktop/project/images/read_crop.jpg')
cap = cv2.VideoCapture('C:/Users/S.S.Aludeniya/Desktop/project/images/ori.3gp')

##filter object by color
def filterImage(image):
    image = cv2.resize(image,(600,336))
    img_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0,120,150])
    upper_bound = np.array([255,255,255])
    mask = cv2.inRange(img_hsv,lower_bound,upper_bound)
    res = cv2.bitwise_and(image,image,mask=mask)
        
    return mask,res

def getSkeleton(mask):
    size = np.size(mask)
    skel = np.zeros(mask.size,np.uint8)
    skel  = cv2.resize(skel,(600,336))

    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
     
    while( not done):
        eroded = cv2.erode(mask,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(mask,temp)
        skel = cv2.bitwise_or(skel,temp)
        mask = eroded.copy()
        cv2.waitKey(1000) 
        zeros = size - cv2.countNonZero(mask)
        if zeros==size:
            done = True
    return skel
    
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
    index=[0,220,0,300]
    z=0
    for x in range(len(b[300])):
        if(b[300][x]== 255):
            index[0]=x
            break
    z=0
    for x in range(len(b[300])):
        if(b[300][x]== 255):
            index[2]=x
            break
    angle = math.degrees(math.atan(abs((index[3]-index[1])/(index[0]-index[2]))))
    return angle
    
def calAngleRead(image):
    b,g,r= cv2.split(image)
    g = np.array(g)
    z = 0
    index=[0,220,0,300]
    for x in range(len(r[220])):
        if(r[220][x]== 255):
            index[0]=x
            break

    for x in range(len(r[350])):
        if(r[350][x]== 255):
            index[2]=x
            break

    if (index[0]>index[2]):
        if((index[0]-index[2])!=0):
            angle  =math.degrees(math.atan(abs((index[3]-index[1])/(index[0]-index[2]))))
            return angle
        else:
            return 1.0
    else:
        if((index[0]-index[2])!=0):
            angle=math.degrees(math.atan(abs((index[3]-index[1])/(index[0]-index[2]))))
            angle = 180.0 - angle 
            return angle
        else:
            return 1.0
        

def calReading(res_main,main_angle):
    ##take frame of video

    while(1):
        ret,frame = cap.read()
        image,res_read=filterImage(frame)
        edges_read = drawLines(res_read,[0,0,255])
        res_read = dilateImage(res_read,1)
        res_read = erodeImage(res_read,5)
    ##result for display    
        result = cv2.bitwise_or(res_main,res_read)

        read_angle = calAngleRead(res_read)
        
        angle = (180.0-(main_angle+read_angle))
        scale_angle = (500.0/(180-(main_angle*2)))*angle
        
        if(scale_angle < 0):
            sacle_angle = 0.0
            print('Reading : ',scale_angle) 
        elif(scale_angle >500):
            sacle_angle = 0.0
            print('Reading : ',scale_angle) 
        else:
            print('Reading : ',scale_angle)
            
        cv2.imshow('Frame',result)

        k=cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    
def main():
##    mask_main,res_main = filterImage(image_main)
##    edges_main = drawLines(res_main,[255,0,0])
##    res_main = dilateImage(res_main,1)
##    res_main = erodeImage(res_main,4)
    
##    calReading(res_main,calAngleInit(res_main))
    while(1):
        res,frame = cap.read()
        mask,res = filterImage(frame)
        cv2.imshow('frame',mask)
        k=cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
   
if __name__ == '__main__':
    main()
    
cv2.waitKey(0)
cv2.destroyAllWindows()


