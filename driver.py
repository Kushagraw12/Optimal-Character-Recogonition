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

# MODEL 1
#cmodel=Sequential()

#cmodel.add(Conv2D(32,(3, 3),activation='relu',input_shape=(28,28,1)))
#cmodel.add(MaxPool2D((2, 2)))
#cmodel.add(Dropout(0.2))

#cmodel.add(Conv2D(64,(3,3),activation='relu'))
#cmodel.add(MaxPool2D((2, 2)))
#cmodel.add(Dropout(0.2))

#cmodel.add(Conv2D(128,(3, 3),activation='relu'))
#cmodel.add(MaxPool2D((2, 2)))
#cmodel.add(Dropout(0.2))

#cmodel.add(Conv2D(128,(3, 3),activation='relu'))
#cmodel.add(MaxPool2D((2, 2)))
#cmodel.add(Dropout(0.2))

#cmodel.add(Conv2D(512,(1, 1),activation='relu'))
#cmodel.add(MaxPool2D((1, 1)))
#cmodel.add(Dropout(0.2))

#cmodel.add(Flatten())


#cmodel.add(Dense(256,activation='relu'))

#cmodel.add(Dense(128,activation='relu'))

#cmodel.add(Dense(64,activation='relu'))

#cmodel.add(Dense(62,activation='softmax'))


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

he = []
lab = []
test = []
for i in range(6284, 10000):
    test.append(io.imread("./test/test/"+str(i)+".Bmp",as_gray=True))
    he.append(i)
    lab.append(str(i).split('.')[0])

test_img = np.array([cv2.resize(image,(28,28)) for image in test])
test_img = test_img[:,:,:,np.newaxis]
print(test_img.shape, "SHAPE")


predictions = model.predict(test_img)

predictions = np.argmax(predictions, axis=1)


#ans = []
for x in range(len(predictions)):
    #ans.append(mapp.get(predictions[x]))
    print(mapp.get(predictions[x]), " PREDICTED FOR ", he[x])
    

#if len(ans) < len(he):
 #   qq = len(ans)
#else:
#    qq = len(he)
#for k in range(qq):
#    print(ans[k], " PREDICTED FOR ", he[k])


cv2.destroyAllWindows()
