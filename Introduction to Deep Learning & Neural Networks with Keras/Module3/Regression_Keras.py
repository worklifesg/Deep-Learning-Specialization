##Lab3_Regression_with_Keras

# Sequential model with Dense layer
# ReLU activation function, Adam - optimizer and loss - MSE

import numpy as np
import pandas as pd 
from tensorflow import keras

## Data read and checking if it is clean 
df=pd.read_csv('D:\Py_Coursera\Deep Learning IBM AI Specialization\Deep Learning with Keras\Labs\concrete_data.csv')
with open('DL_Reg_Keras.txt','a') as f:
    print(df.head(),file=f)
    print(df.shape,file=f)
    print(df.describe(),file=f)
    print(df.isnull().sum(),file=f)

## Dividing the data into predictors and target variables for simplicity
concrete_data_columns = df.columns
predictors = df[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = df['Strength'] # Strength column

# Normalizing the predictors and target - normalize the data by substracting the mean and dividing by the standard deviation.
predictors_norm = (predictors - predictors.mean()) / predictors.std()
n_cols = predictors_norm.shape[1] # number of predictors

with open('DL_Reg_Keras.txt','a') as f:
    print(predictors.head(),file=f)
    print(target.head(),file=f)
    print(predictors_norm.head(),file=f)
    print('No. of predictors: ',n_cols,file=f)

## Use Sequential model as the netowkr consists of linear stack of layers
# Use 'Dense' layer type

from keras.models import Sequential
from keras.layers import Dense


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,))) # Adding neurons to hidden layer 1
    model.add(Dense(50, activation='relu')) # Adding neurons to hidden layer 2
    model.add(Dense(1)) # Adding neurons to output
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error') # -Adam = Scholastic Gradient Descent, MSE - between predicted value and T
    return model

# build the model
model =regression_model()

# fit the model to train the dataset
# We will leave out 30% of the data for validation and we will train the model for 100 epochs.

model_train=model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

with open('DL_Reg_Keras.txt','a') as f:
    print(model_train,file=f)
