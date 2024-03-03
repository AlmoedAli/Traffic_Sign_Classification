import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout

fileOriginal= 'Dataset/Train/'

def reshapeImage():
    k= 0
    for i in range (0, 6):
        SubPath = fileOriginal +  str(i)+ "/"
        for j in range (1, 61):
            SubSubPath= SubPath + str(i) + "_ ("+ str(j)+ ").jpg"
            image= cv.imread(SubSubPath)
            image= cv.resize(image, (64, 64))
            cv.imwrite(SubSubPath, image)
            k+= 1     
    return k    
    
    
def generateData():
    X= []
    y= []
    for i in range (0, 6):
        SubPath = fileOriginal +  str(i)+ "/"
        for j in range (1, 61):
            SubSubPath= SubPath + str(i) + "_ ("+ str(j)+ ").jpg"
            image= cv.imread(SubSubPath)
            X.append(image)
            y.append(i)
    X= np.array(X)
    y= np.array(y)
    return X, y

def convertOneHot(y):
    y= y.reshape(-1, 1)
    yConvert= np.zeros((y.shape[0], np.max(y)+ 1))
    for i in range (y.shape[0]):
        yConvert[i, y[i, 0]]= 1
    return yConvert

def ModeCNN():
    model= Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation= "relu", input_shape= (64, 64, 3), padding= "same"))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation= "relu", padding= "same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation= "relu", padding= "same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate= 0.2))
    
    model.add(Flatten())
    model.add(Dense(256, activation= "relu"))
    model.add(Dense(128, activation= "relu"))
    model.add(Dense(32, activation= "relu"))
    model.add(Dense(6, activation= "softmax"))
    
    model.summary()
    
    model.compile(optimizer= "adam", loss= 'categorical_crossentropy', metrics= ['accuracy'])
    return model



if __name__== "__main__":
    total = reshapeImage()
    X, y= generateData()
    XTrain, XTest, yTrain, yTest= train_test_split(X, y, test_size= 0.1, random_state= 42)
    XTrain= XTrain/ 255
    XTest= XTest/ 255
    yTrainConvert= convertOneHot(yTrain)
    yTestConvert= convertOneHot(yTest)
    print(yTrainConvert.shape)
    model= ModeCNN()
    history= model.fit(XTrain, yTrainConvert, batch_size= 100, epochs= 5, validation_data= (XTest, yTestConvert))
    model.save("mode.h5")
