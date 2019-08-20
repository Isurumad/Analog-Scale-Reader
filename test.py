import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

image_main = cv2.imread('C:/Users/S.S.Aludeniya/Desktop/project/images/initnew.jpg')
image_read = cv2.imread('C:/Users/S.S.Aludeniya/Desktop/project/images/read_crop.jpg')
cap = cv2.VideoCapture('ori.3gp')

##filter object by color
def filterImage(image):
    image = cv2.resize(image,(600,336))
    img_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0,120,150])
    upper_bound = np.array([255,255,255])
    mask = cv2.inRange(img_hsv,lower_bound,upper_bound)
    res = cv2.bitwise_and(image,image,mask=mask)
        
    return image,mask,res

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

    return image,edges

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
    b = np.array(b)
    z = 0
    index=[0,160,0,260]
    for x in range(len(r[160])):
        if(r[160][x]== 255):
            index[0]=x
            break

    for x in range(len(r[260])):
        if(r[260][x]== 255):
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
        

def calReading(main_angle):
    ##take frame of video

    while(1):
        ret,frame = cap.read()
        image,mask,res_read=filterImage(frame)
        edges_read,edges = drawLines(res_read,[0,0,255])
        
        res_read = erodeImage(res_read,2)
        res_read = dilateImage(res_read,1)
    
        read_angle = calAngleRead(res_read)
        angle = (180.0-(main_angle+read_angle))
        scale_angle = math.ceil((500.0/(180-(main_angle*2)))*(angle))
      
        
        if(scale_angle <= 0):
            print('Reading : ',0.0) 
        else:
            print('Reading : ',(scale_angle))
            
        cv2.imshow('Image',image)
        cv2.imshow('Frame',edges)

        k=cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

def drawRefLine(image):
    b,g,r = cv2.split(image)
    g = np.array(g)
    for x in range(len(g[160])):
        g[160][x]=255

    for x in range(len(g[260])):
        g[260][x]=255

    image = cv2.merge([b,g,r])
    return image
def test():
##    mask_main,res_main = filterImage(image_main)
##    edges_main = drawLines(res_main,[255,0,0])
##    res_main = dilateImage(res_main,1)
##    res_main = erodeImage(res_main,4)
    
##    calReading(res_main,calAngleInit(res_main))
    while(1):
        res,frame = cap.read()
        mask,res = filterImage(frame)
        edges = drawLines(res,[255,0,0])
        dilate =dilateImage(erodeImage(edges,2),1)
        calAngleRead(drawRefLine(dilate))
        cv2.imshow('frame',drawRefLine(dilate))
        k=cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

   
    
def main():
    calReading(55.02)
    
if __name__ == '__main__':
    main()
    
cv2.waitKey(0)
cv2.destroyAllWindows()


