# Building Deep Learning Models with TensorFlow

Please note: Most of the programs of the course are done using TensorFlow 1.X and 
now the whole main focus has been shifted to TensorFlow 2.X.

Some general program in Module 1 are done by:

```python      
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior() 
```

But the above technique may not be available after certain time, so instead of course
lab programs, general programs based on CNN, RNN, and Autocoders are provided for both 
versions to learn and understand the general shift from V1 to V2.

All general differences between both versions is given in TensorFlowV1V2.md file


* Module 1 introduces the concept of tensorflow, deep learning and deep neural netoworks. It consists of certain lab exercises such as usage of tensorflow in: general mathematical operations, Linear Regression, and Logistic Regression.
* Module 2 covers building blocks of Convolution Neural Networks (CNN) which as supervised learning models. Programs related to MNIST data images is written using both versions of tensorflow for better understanding. 

## Table of contents
* [Introduction to TensorFlow](#introduction-to-tensorflow)
* [Convolution Neural Networks](#convolution-neural-networks)

### Introduction to TensorFlow

* Python files: 
  * Lab1_TensorFlow_HelloWorld.py
  * Lab1_TensorFlow_LinearRegression.py
* Output file: 
  * Lab1_TensorFlow_HelloWorld
  * Lab1_TensorFlow_LinearRegression.txt and [Graphs](https://github.com/worklifesg/Deep-Learning-Specialization/blob/master/Building%20Deep%20Learning%20Models%20with%20TensorFlow/Module1/Lab1_TensorFlow_LinearRegression_Graphs.pdf) 
* Dataset Files:
  * Linear Regression - [Fuel Consumption CO2](https://github.com/worklifesg/Deep-Learning-Specialization/blob/master/Building%20Deep%20Learning%20Models%20with%20TensorFlow/Module1/FuelConsumptionCo2.csv)


### Convolution Neural Networks

RAW files means the concept of weights and biases is used to initialize each layer and used to build and train the model.
APIs file means layers/estimators are used to nitialize each layer and used to build and train the model.

* Jupyter files: 
  * [TensorFlow_V1_CNN_RawVersion](https://github.com/worklifesg/Deep-Learning-Specialization/blob/master/Building%20Deep%20Learning%20Models%20with%20TensorFlow/Module2/TensorFlow_V1_CNN_RawVersion.ipynb)
  * [TensorFlow_V1_CNN_APIsVersion](https://github.com/worklifesg/Deep-Learning-Specialization/blob/master/Building%20Deep%20Learning%20Models%20with%20TensorFlow/Module2/TensorFlow_V1_CNN_APIsVersion.ipynb)
  * [TensorFlow_V2_CNN_APIsVersion](https://github.com/worklifesg/Deep-Learning-Specialization/blob/master/Building%20Deep%20Learning%20Models%20with%20TensorFlow/Module2/TensorFlow_V2_CNN_APIsVersion.ipynb)
  * [TensorFlow_V2_CNN_RawVersion](https://github.com/worklifesg/Deep-Learning-Specialization/blob/master/Building%20Deep%20Learning%20Models%20with%20TensorFlow/Module2/TensorFlow_V2_CNN_RawVersion.ipynb)
* Dataset:
  * [MNIST](http://yann.lecun.com/exdb/mnist/)
