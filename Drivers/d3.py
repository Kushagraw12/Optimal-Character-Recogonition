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


# MODEL 2
model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1),data_format='channels_last'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

#model.add(Dense(512,activation='relu'))

#model.add(Dense(256,activation='relu'))

model.add(Dense(128,activation='relu'))

model.add(Dense(62,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.load_weights('./weights/z-1-weights-0.8679.h5')

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
ans = []
#for i in range(10115, 10121):
#    test.append(io.imread("./Test/"+str(i)+".Bmp",as_gray=True))
#    he.append(i)
#    lab.append(str(i).split('.')[0])

vid = cv2.VideoCapture(0)

while(cv2.waitKey(1) != ord('q')): #and (start - end != 3)):
    ret, frame = vid.read()

    cv2.imshow('frame', frame)

   # if cv2.waitKey(1) & 0xFF == ord('q'): 
    #    break

    fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    test.append(fr)
    
    test_img = np.array([cv2.resize(image,(28,28)) for image in test])
    test_img = test_img[:,:,:,np.newaxis]
    #print(test_img.shape, "SHAPE")


    predictions = model.predict(test_img)

    predictions = np.argmax(predictions, axis=1)


   # print(mapp.get([x for x in predictions]), " : PREDICTED")
    for x in predictions:
        ans.append(mapp.get(x))
        print(mapp.get(x), " : PREDICTED")

   # if len(ans) < len(he):
   #     qq = len(ans)
   # else:
   #     qq = len(he)
    #for k in range(len(ans)):
     #   print(ans[k], " PREDICTED")

print("ENDING")
coun = 0
for cc in ans:
    if(cc == "k" or cc == 'K'):
        coun += 1

print("Correct Ratio: ", coun, " : ", len(ans))
cv2.destroyAllWindows()

