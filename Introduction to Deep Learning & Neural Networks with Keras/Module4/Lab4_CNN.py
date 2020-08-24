# Lab4_CNN
# In this example, we will use MNIST dataset and evaluate the results using CNN
# Two CNN models, one model (model) with single convolution and pooling layer and second model (model1) with two convolution and pooling layers. 
# Importing libraries

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

## for CNN - libraries:

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten

## Dataset

from keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# The MNIST database contains 60,000 training images and 10,000 testing 
# images of digits written by high school students and employees of the United States Census Bureau.

X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')

# Normalizing pixel values [0,255] to [0 1]

X_train=X_train/255
X_test=X_test/255

# using one hot encoding to have categorical outputs for TARGET

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

num_classes=y_test.shape[1]
with open('DL_CNN_Keras.txt','a') as f:
    print(num_classes,file=f)

def convolutional_model():
    model=Sequential()
    model.add(Conv2D(16,(5,5),strides=(1,1),activation='relu',input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model=convolutional_model()

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=200,verbose=0)

scores=model.evaluate(X_test,y_test,verbose=0)

def convolutional_model1():
    model1=Sequential()
    model1.add(Conv2D(16,(5,5),strides=(1,1),activation='relu',input_shape=(28,28,1)))
    model1.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model1.add(Conv2D(8,(5,5),activation='relu'))
    model1.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model1.add(Flatten())
    model1.add(Dense(100,activation='relu'))
    model1.add(Dense(num_classes,activation='softmax'))

    model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model1

model1=convolutional_model1()

model1.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=200,verbose=0)

scores1=model1.evaluate(X_test,y_test,verbose=0)

with open('DL_CNN_Keras.txt','a') as f:
    print("Using single CNN and pooling layer: \n Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100),file=f)
    print("Using two CNN and pooling layers: \n Accuracy: {} \n Error: {}".format(scores1[1], 100-scores1[1]*100),file=f)

