import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU
import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle

DataDir = "/Users/graysonhalliday/Documents/greetingNeuralNet"
Categories = ["Juan","Graeme","Grayson","Liam", "Spencer"]
IMG_SIZE = 200

trainingData = []

def createTrainingData():
    for person in Categories:
        print(person)
        curPath = os.path.join(DataDir,person)
        classNum = Categories.index(person)
        for img in os.listdir(curPath):
            try:
                if (img == ".DS_Store"):
                    continue
                img_array = cv2.imread(os.path.join(curPath,img),cv2.IMREAD_COLOR)
                resizedImgArray = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                trainingData.append([resizedImgArray,classNum])
            except Exception as e:
                print("Exception")
                pass


createTrainingData() 
random.shuffle(trainingData)

X = []
Y = []

for features, label in trainingData:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE, 3)
Y = keras.utils.to_categorical(Y, 5)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

X = X/255.0

model = Sequential()

model.add(Conv2D(16, (6,6), input_shape=X.shape[1:]))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (4,4), input_shape=X.shape[1:]))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), input_shape=X.shape[1:]))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=['accuracy'])

model.fit(X,Y,batch_size=20, validation_split=0, epochs=10)

model.save('greeting_model')

testPath = os.path.join(DataDir,"testingData")
T = []
for testImage in os.listdir(testPath):
    print(testImage)
    if (testImage == ".DS_Store"):
        continue
    img_array = cv2.imread(os.path.join(testPath,testImage),cv2.IMREAD_COLOR)
    resizedImgArray = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    T.append(resizedImgArray)

T = np.array(T).reshape(-1,IMG_SIZE,IMG_SIZE, 3)

print(model.predict(T))