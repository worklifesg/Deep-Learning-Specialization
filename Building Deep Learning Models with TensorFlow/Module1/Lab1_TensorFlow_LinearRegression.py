## Here we will use tensorflow with Linear Regression
# Recap: Defining a linear regression in simple terms, is the approximation of a linear model 
# used to describe the relationship between two or more variables.
# When more than one independent variable is present the process is called multiple linear regression.
# When multiple dependent variables are predicted the process is known as multivariate linear regression.

#Importing libraries

import numpy as np
import pandas as pd 

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Disabling version 2 as eagar execution gives few codes error.

plt.rcParams['figure.figsize']=(8,6)

# in linear regression, we always want to adjust slope and intercept to the data the best way possible.

#Data read: using same example that we used in machine learning course: FuelConsumptionCo2.csv

df=pd.read_csv('D:\Py_Coursera\Deep Learning IBM AI Specialization\Building Deep Learning Models with TensorFlow\Labs\FuelConsumptionCo2.csv')
with open('Lab1_TensorFlow_LinearRegression.txt','a') as f:
    print(df.head(),file=f)
    print('\n',file=f)

# We want to predict CO2 emissions of car based on their engine size
train_x=np.asanyarray(df[['ENGINESIZE']])
train_y=np.asanyarray(df[['CO2EMISSIONS']])

# Intialize variables using tensorflow with random values
a=tf.Variable(20.0) #slope
b=tf.Variable(30.2) #intercept
y=a*train_x+b #line equation

# We have to define tow functions:
    ## Loss: MSE - minimize the square of the predicted values minus target value
    ## Optimizer method : gradient descent method - LR parameter corresponds to speed with which the optimizer should learn. Can change the method

loss = tf.reduce_mean(tf.square(y-train_y)) # reduce_mean - finds mean of tensor, square - MSE
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.05, use_locking=False, name='GradientDescent')# LR - 0.05 using gradient descent optimizer
#disable_eager_execution()
# To train the graph, we need to minimize the error function of our optimizer using tf.minimize()
train_op=optimizer.minimize(loss)

# To run the session for varaiables:
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

loss_values = [] #initialize loss values empty matrix
train_data = [] #initilize train data empty matrix
for step in range(100):
    _, loss_val, a_val, b_val = sess.run([train_op, loss, a, b]) #running session to displya values
    loss_values.append(loss_val)
    if step % 5 == 0:
        with open('Lab1_TensorFlow_LinearRegression.txt','a') as f:
            print(step, loss_val, a_val, b_val,file=f)
            train_data.append([a_val, b_val])

#plot the cost function for linear regression using gradient descent of LR=0.05
plt.figure()
plt.plot(loss_values,'ro')

# visualize coefficient and intercept fit the data
plt.figure()
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(train_x)
    line = plt.plot(train_x, f_y)
    plt.setp(line, color=(cr,cg,cb))


plt.plot(train_x, train_y, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])

#Display plot
plt.show()