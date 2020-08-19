##Lab3_Classification_with_Keras

# Sequential model with Dense layer
# ReLU activation function, Adam - optimizer and loss - MSE

# After saving model we can import it without losing computation memory and time
#from keras.models import load_model
#pretrained_model = load_model('classification_model.h5')

import numpy as np
import pandas as pd 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt 

## Data read and checking if it is clean (import dataset from MNIST dataset library)

from keras.datasets import mnist

(X_train,y_train),(X_test,y_test) = mnist.load_data()

# The MNIST database contains 60,000 training images and 10,000 testing 
# images of digits written by high school students and employees of the United States Census Bureau.

with open('DL_Class_Keras.txt','a') as f:
    print(X_train.shape,file=f)

# visualize the first image
plt.figure()
plt.imshow(X_train[0])
plt.show()

# Data cleaning and fallten images to 1D vector

num_pixels=X_train.shape[1]*X_train.shape[2]

X_train=X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
X_test=X_test.reshape(X_test.shape[0],num_pixels).astype('float32')

# Normalizing pixel values [0,255] to [0 1]

X_train=X_train/255
X_test=X_test/255

# using one hot encoding to have categorical outputs for TARGET

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

num_classes=y_test.shape[1]
with open('DL_Class_Keras.txt','a') as f:
    print(num_classes,file=f)

def classification_model():
    model=Sequential()
    model.add(Dense(num_pixels,activation='relu',input_shape=(num_pixels,)))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model=classification_model()
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,verbose=2)

scores=model.evaluate(X_test,y_test,verbose=0)

model.save('classification_model.h5')
