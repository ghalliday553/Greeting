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
TestCategories = ["JuanTest","GraemeTest","GraysonTest","LiamTest","SpencerTest"]
IMG_SIZE = 500

trainingData = []

def testModel():
    testSizes = [0,0,0,0,0]
    testResults = [0,0,0,0,0]

    curPath = os.path.join(DataDir,"testingData")
    testData = []
    for testPerson in os.listdir(curPath):
        #print(testPerson)
        if (testPerson == ".DS_Store"):
                continue
        testPath = os.path.join(curPath,testPerson)
        classNum = TestCategories.index(testPerson)
        for testImage in os.listdir(testPath):
            if (testImage == ".DS_Store"):
                continue
            testSizes[classNum]+=1
            print(testImage)
            img_array = cv2.imread(os.path.join(testPath,testImage),cv2.IMREAD_COLOR)
            resizedImgArray = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            testData.append([resizedImgArray,classNum])

    XTest = []
    YTest = []

    for features, label in testData:
        XTest.append(features)
        YTest.append(label)

    XTest = np.array(XTest).reshape(-1,IMG_SIZE,IMG_SIZE, 3)
    YTest = keras.utils.to_categorical(YTest, 5)

    XTest = XTest/255.0

    #print(YTest)
    #print(testSizes)
    predictions = model.predict(XTest)
    #print(predictions)

    for index, testPrediction in enumerate(predictions):
        detPerson = 0
        actPerson = 0
        print("index is " + str(index))
        for detIndex, detectedPerson in enumerate(testPrediction):
            if detectedPerson > 0.5:
                detPerson = detIndex
                #print("det person is " + str(detIndex))
                break
        #print("index after is" + str(index))
        #print(YTest[index])
        for actIndex, actualPerson in enumerate(YTest[index]):
            if actualPerson == 1:
                actPerson = actIndex
                #print("act person is " + str(actIndex))
                break
        if detPerson == actPerson:
            testResults[actPerson]+=1
    
    for index, results in enumerate(testResults):
        print(Categories[index])
        print("Size: " + str(testSizes[index]) + " Got: " + str(testResults[index]) + " For: " + str(results/testSizes[index]))


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

model.add(Conv2D(16, (4,4), input_shape=X.shape[1:]))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), input_shape=X.shape[1:]))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(128, (3,3), input_shape=X.shape[1:]))
#model.add(LeakyReLU(alpha=0.1))
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

#model.add(Dense(128))
#model.add(Dense(128))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=['accuracy'])

model.fit(X,Y,batch_size=20, validation_split=0, epochs=15)

model.save('greeting_model')
testModel()