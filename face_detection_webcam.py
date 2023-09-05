# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 21:03:25 2023

@author: SATYAKI
"""

import cv2,time,sys


def face_detection_webcam()->None:
    """Detect face using default webcam"""
    #Loading the face detection model with opencv
    face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    #img = cv2.imread('Sample-01.jpg')
    
    # Capturing the video 
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        sys.exit("Camera not opend ! Exiting ...... ")
    
    
    count=0
    frame_rate = 2
    prev = 0
    
    
    while True:
        time_elapsed = time.time() - prev
        res,frame = cap.read()
        
        if not res:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        #convert to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2d array of face square 
        faces = face_cascade.detectMultiScale(gray,1.1,4)
        
        #Drawing rectangle on faces
        for(x,y,h,w) in faces:
            cv2.rectangle(frame, (x,y), (x+h,y+w),(0,0,255),2)
            
        #Controlling the frame rate
        if time_elapsed > 1./frame_rate:
            prev = time.time()
            
            #Saving the images in gray sacle format 
            #Only the focused area is saved
            for (x1,y1,h1,w1) in faces:
                cv2.imwrite('./images/' + str(count) + '.pgm',
                            gray[y1:y1+w1,x1:x1+h1],
                            [cv2.IMWRITE_PXM_BINARY,0]
                            )
                count += 1
                
        # Showing the video capture         
        cv2.imshow('Video Capture !', frame)
        
        # Exiting the function with 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
   
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    face_detection_webcam()