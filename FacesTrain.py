# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 19:08:44 2020

@author: ritap
"""

import os
import numpy as np
import pandas as pd
import keras.backend as K
import tensorflow as tf
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')
face_cascade = cv2.CascadeClassifier('D:/Work/Data_Science/Projects/FaceDetection/files/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
train_X = list()
train_y = list()
labels = {}
current_id = 0
for folder in os.listdir(image_dir):
    labels[current_id] = folder
    #print(folder)
    for file in os.listdir(os.path.join(image_dir,folder)): 
        img = Image.open(os.path.join(os.path.join(image_dir,folder),file)).convert('L')
        ima = img.resize((300,200))
        img = np.array(img)
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5)
        for x,y,w,h in faces:
            #print(x,y,w,h)
            #cv2.rectangle(frame, (x,y), (x+w, y+h),(50,100,250), 3)
            roi_gray = img[y:y+h, x:x+w]
            #roi_color = frame[y:y+h, x:x+w] 
            train_X.append(roi_gray)
            train_y.append(current_id)
    current_id+=1

with open('labels.pickle', 'wb') as f:
    pickle.dump(labels, f)
    

recognizer.train(train_X, np.array(train_y))
recognizer.save('trainer.yml')



# =============================================================================
# train_X = np.array(train_X)
# train_X[i:i+5000] = train_X[i:i+5000]/255    
# np.save('train_X.npy', train_X)
# train_y = np.array([0 for x in range(12500)] + [1 for x in range(12500)])
# np.save('train_y.npy', train_y)
# =============================================================================
    
