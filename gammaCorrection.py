import cv2
import numpy as np

def gammaCorrection(image,gamma):

    invGamma = 1.0/gamma
    gammaCorrection = np.array(image)

    for x in range(len(gammaCorrection)):
        for y in range (len(gammaCorrection[x])):
            gammaCorrection[x][y] = ((gammaCorrection[x][y]/255.0)** imvGamma)*255

    return gammaCorrection


image = cv2.imread('H:\CS314\Day 6\images\fire\fire.jpg',0)
image = gammaCorrection(image,1.0)

cv2.imshow('gammaCorrection',image)
            

    

##def gammaCorrection(image,gamma):
##    
##    imvGamma = 1.0/gamma
##    table = np.array([((i/255.0)** imvGamma)*255
##                      for i in np.arange(0,256)]).astype("unit8")
##
##    return cv2.LUT(image,table)
##
##image = cv2.imread('H:\CS314\Day 6\images\fire\fire.jpg')
##
##image,table = gammaCorrection(image,1.0)
##
##cv2.imshow('GammaCorrection',image)
##cv2.destroyAllWindows()




