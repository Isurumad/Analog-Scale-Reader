import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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

def getSkeleton(image,mask):
    size = np.size(image)
    skel = np.zeros(image.size,np.uint8)
    skel  = cv2.resize(skel,(600,336))

    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
     
    while( not done):
        eroded = cv2.erode(mask,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(mask,temp)
        skel = cv2.bitwise_or(skel,temp)
        mask = eroded.copy()
        cv2.waitKey(2) 
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
    
def calAngleRead(image):
    b,g,r = cv2.split(image)
    r=np.array(r)
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
            angle  =math.degrees(math.atan(abs((index[3]-index[1])/(index[2]-index[0]))))
            return angle
        else:
            return 1.0
    else:
        if((index[0]-index[2])!=0):
            angle=math.degrees(math.atan(abs((index[3]-index[1])/(index[2]-index[0]))))
            angle = 180.0 - angle 
            return angle
        else:
            return 1.0
        
def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0
    color = (0,255,0)
    thickness = cv2.FILLED
    margin = 10

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
    
def calReading(main_angle):
    ret = True
    while(ret):
        ret,frame = cap.read()
    
        image,mask,filter_image=filterImage(frame)
        image_edge,edges = drawLines(filter_image,[0,0,255])
            
        res_read = erodeImage(image_edge,2)
            
        skel = getSkeleton(res_read,mask)
        line_image = drawRefLine(res_read)
            
        read_angle = calAngleRead(res_read)
        angle = (180.0-(main_angle+read_angle))
        reading = 0
        if angle <=13:
            reading = math.ceil((100.0/(13.58))*(angle))
        elif angle <= 29:
            reading = (math.ceil((100.0/(16.09))*(angle-13)))+100
        elif angle <= 47:
            reading = (math.ceil((100.0/(17.57))*(angle-29)))+200
        elif angle <= 60:
            reading = (math.ceil((100.0/(13.82))*(angle-47)))+300
        else:
            reading = (math.ceil((100.0/(9.7))*(angle-59)))+400
            
        if(reading <= 0):
            reading = 0.0 

        draw_label(image,str(reading),(50,75), (25,25,25))
        cv2.imshow('Image',image)
        cv2.imshow('Frame',line_image)
        cv2.imshow('Skeleton',skel)

        k=cv2.waitKey(10) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

def drawRefLine(image):
    b,g,r = cv2.split(image)
    for x in range(len(g[150])):
        g[150][x]=255

    for x in range(len(g[260])):
        g[260][x]=255
    image = cv2.merge([b,g,r])
    return image

def test():
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
    calReading(56.2)
    
if __name__ == '__main__':
    main()
    
cv2.waitKey(0)
cv2.destroyAllWindows()


