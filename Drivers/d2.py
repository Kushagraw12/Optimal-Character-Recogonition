import pandas
import numpy as np
from keras.preprocessing import image
from keras.layers import Conv2D,Flatten, Dense, Dropout, MaxPool2D
from keras.optimizers import Adam
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import Adam
import scipy.misc
import tensorflow as tf
import cv2
from subprocess import call
import os
import matplotlib.pyplot as plt
from keras.layers import Conv2D,Dense,MaxPool2D,Dropout,Flatten
from keras.models import Sequential
from skimage import io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import time
from timeit import default_timer as timer


cmodel=Sequential()

cmodel.add(Conv2D(32,(3, 3),activation='relu',input_shape=(28,28,1)))
cmodel.add(MaxPool2D((2, 2)))
cmodel.add(Dropout(0.2))

cmodel.add(Conv2D(64,(3,3),activation='relu'))
cmodel.add(MaxPool2D((2, 2)))
cmodel.add(Dropout(0.2))

cmodel.add(Conv2D(128,(3, 3),activation='relu'))
cmodel.add(MaxPool2D((2, 2)))
cmodel.add(Dropout(0.2))

#cmodel.add(Conv2D(128,(3, 3),activation='relu'))
#cmodel.add(MaxPool2D((2, 2)))
#cmodel.add(Dropout(0.2))

#cmodel.add(Conv2D(512,(1, 1),activation='relu'))
#cmodel.add(MaxPool2D((1, 1)))
#cmodel.add(Dropout(0.2))

cmodel.add(Flatten())


#cmodel.add(Dense(256,activation='relu'))

cmodel.add(Dense(128,activation='relu'))

cmodel.add(Dense(64,activation='relu'))

cmodel.add(Dense(62,activation='softmax'))

cmodel.summary()
cmodel.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

cmodel.load_weights('../Weights/z-1-weights-0.6181.h5')

#img1 = cv2.imread('s1.png',0)
#rows,cols = img_str.shape

#img1 = image.img_to_array(img1) / 255.0

#img1 = io.imread("s1.png",as_gray=True) #image.load_img("s1.png", color_mode = 'rgb', target_size = [28, 28, 1])
#img1 = image.img_to_array(img1) / 255.0
#img1 = cv2.resize(img1, (28, 28)).flatten()
#img = np.array(img1)

#pred = cmodel.predict(img)
#print(pred)

mapp={}
a='abcdefghijklmnopqrstuvwxyz'
count=0
for x in range(10):
    mapp[x]=count
    count+=1
for y in a:
    mapp[count]=y.upper()
    count+=1
for y in a:
    mapp[count]=y
    count+=1

#he = []
#lab = []
test = []
#for i in range(10115, 10121):
#    test.append(io.imread("./Test/"+str(i)+".Bmp",as_gray=True))
#    he.append(i)
#    lab.append(str(i).split('.')[0])

vid = cv2.VideoCapture(0)
sta = time.time()
start = timer()
en = sta
end = timer()

while(cv2.waitKey(1) != ord('q') and (start - end != 3)):
    ret, frame = vid.read()

    cv2.imshow('frame', frame)

   # if cv2.waitKey(1) & 0xFF == ord('q'): 
    #    break

    fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    test.append(fr)
    
    test_img = np.array([cv2.resize(image,(28,28)) for image in test])
    test_img = test_img[:,:,:,np.newaxis]
    #print(test_img.shape, "SHAPE")


    predictions = cmodel.predict(test_img)

    predictions = np.argmax(predictions, axis=1)


    ans = []
    for x in predictions:
        ans.append(mapp.get(x))

   # if len(ans) < len(he):
   #     qq = len(ans)
   # else:
   #     qq = len(he)
    for k in range(len(ans)):
        print(ans[k], " PREDICTED AT", end)

    end = time.time()


cv2.destroyAllWindows()

