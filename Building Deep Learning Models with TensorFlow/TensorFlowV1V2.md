## TensorFlow versions 1.X and 2.X 

This markdown is related to differences between TensorFlow version 1.X and 2.X. I will discuss first general high level differences and then changes done for Convolution Neural Netowrks (CNNs), Recurrent Neural Networks (RNNs) and Autoencoders. The programs for these learning models are also explained in their respective Jupyter notebooks as well.

### *Easy to use libraries/packages*

Earliar in version 1.X, there were libraries such as Contrib, layers, Keras or estimators that were used to build a model and had some confusion about their utility and usage. Therefore new version 2.X promotes more of 'Keras' for model built and 'Estimators' for scaled serving. 

So, just focusing on these two APIs, some of the old libraries such as      ```tensorflow.examples.tutorials.mnist ``` will be depreciated in future versions or removed in 2.X and now datasets can be retrieved from Keras library as ```from tensorflow.keras.datasets import mnist```

### *Eagar Execution*

```
Eagar execution is used in tensorflow to create a session to run the computational graph.
```
Eagar execution is enabled by default in 2.X and the reason to enable this function has changed the workflow of writing programs tremendously. In general (in 1.X), writing code is done in two parts: computational graph and creating a session to execute it. This process became tiresome and cumbersome specifically for large datasets and even small error in code can take a lot of time to correct the error and its dependancies in the compuatational graph and session. By enabling this function in 2.X, some functions do not work and had to be removed in version 2. 

For example, ``` tf.placeholder()``` can't work when eagar execution is enabled, therefore it has been removed in 2.X

### *Model Building and Deployment*

In general, we need to build computational graph in 1.X and 2.X doesn't build graph b default. But graphs are good for speed, so in 2.X provides the user to create callable graph using python function ```@tf.function - tf.function()```. This function creates a separate graph for every unique set of input shapes and datatypes. 

```
Why is tf.function() a useful feature in 2.X?

- tf.function()  decorator runs as a single grah via 'Autograph' feature of 2.X. 
- This feature allows to optimize the function and increase portability.
- This function can turn plain Python code into graph
- Any function called after calling tf.function will be executed in graph.
- ONLY use tf.function to decorate high level computations only.
```

### *Data pipeline simplification*

This part has been discussed briefly above in instances. But to summarize some major encounters are:

For Version 1.X:

```
- separate module for Datasets
- to build a model, we have to define placeholders (dummy variables later used to feed the data)
- Many built in APIs to execute mathematical operations (contrib,layers,keras)
```

For Version 2.X:

```
- Use already in-built Datasets from Keras
- no placeholders
- can define our own mathematical operations using tf.math, tf.linalg, etc.
```

### *Summary for Tensorflow 2.X*

```
- Load data by tf.data (Also using keras datasets if using pre-defined datasets)
- Build, train, validate model using Keras or Estimators API
- Can use TensorFlow Hub to import data and wither train from scratch or fine tune
- Run and Debug with Eagar Execution and then use tf.function() for benefits of graphs (if needed)
- Usage of Distribution Strategy API for large machine learning tasks that are easy to distribute and train models on different
  hardware requirements.
- Export to 'SavedModel'
```
