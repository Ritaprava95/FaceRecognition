# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 00:47:30 2020

@author: ritap
"""
import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('D:/Work/Data_Science/Projects/FaceDetection/files/haarcascade_frontalface_alt2.xml')
smile_cascade = cv2.CascadeClassifier('D:/Work/Data_Science/Projects/FaceDetection/files/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
labels = {}
ages = {}
with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)
with open('ages.pickle', 'rb') as f:
    ages = pickle.load(f)
cap = cv2.VideoCapture(0)
w_prev, h_prev = 0, 0
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in faces:
        #print(w)
        if w > w_prev+10 and h > h_prev+10:
            print("Comming Towards")
        elif w < w_prev-10 and h < h_prev-10:
            print("Going Away")
        w_prev, h_prev = w, h
        #print(x,y,w,h)
        cv2.rectangle(frame, (x,y), (x+w, y+h),(50,100,250), 1)
        #cv2.circle(frame, (x+w//2,y+w//2), w//2,(50,100,250), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w] 
        
        #recognize
        label, conf = recognizer.predict(roi_gray)
        if conf>40:
            #print(labels[label])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[label]
            print(name)
            age = str(ages[name])
            color = (255, 255, 255)
            stroke = 1
            cv2.putText(frame, name, (x,y-10), font, .7, color, stroke, cv2.LINE_AA)
            cv2.putText(frame, age, (x+50,y-10), font, .7, color, stroke, cv2.LINE_AA)
        image_item = "myImage.png"
        cv2.imwrite(image_item, roi_gray)
# =============================================================================
#     smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
#     for x, y, w, h in smiles:
#         cv2.rectangle(frame, (x,y), (x+w, y+h),(50,100,250), 1)
# =============================================================================
        
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
 
