# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 07:31:43 2023

@author: SATYAKI
"""

import cv2 as cv
import sys

def face_detection():

    frameno=0
    
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    img = cv.imread('./samples/sample_02.jpg')
    
    
    if img is None:
        sys.exit("Could not open the image file")
        
    img_resize = cv.resize(img, (1080,720))
    gray = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        image=gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(120,120)
        )
    
    
    for (x, y, h, w) in faces:
        cv.rectangle(img_resize, (x, y), (x+h, y+w), (255, 0, 0), 2)
        cv.imwrite('./images/'+ str(frameno) +'.pgm', gray[y:y+w,x:x+h],[cv.IMWRITE_PXM_BINARY,0])
        frameno += 1
        
    cv.imshow('img', img_resize)
    cv.waitKey()
    
    
if __name__ == "__main__":
    face_detection()