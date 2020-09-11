#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info" style="margin-top: 20px">
# <font size = 5><strong>Example: Convolutional Neural Network using TensorFlow (1.15) </strong></font>

# <h3>Convolution Neural Network</h3>
# 
# ![CNN](https://raw.githubusercontent.com/worklifesg/Deep-Learning-Specialization/master/images/CNN.png)

# <h3>MNIST Dataset</h3>
# 
# The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
# 
# It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. 
# 
# ![MNIST](https://raw.githubusercontent.com/worklifesg/Deep-Learning-Specialization/master/images/MnistExamples.png)

# </div>
# <div class="alert alert-success" role="alert">
#   Step I: Import <b><u>Libraries & MNIST</u></b> Data
# </div>

# In[1]:


from __future__ import division, print_function,absolute_import
import tensorflow as tf


# In[2]:


#MNSIT data using tutorials only valid in TensorFlow V1

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)


# </div>
# <div class="alert alert-success" role="alert">
#   Step II: Define Parameters --> <b><u>Training, Network, Graph input</u></b>
# </div>

# In[3]:


#Training parameters

learning_rate=0.001
num_steps=500
batch_size=128
display_step=10


# In[4]:


#Network parameters (MNIST data parameters)

num_input=784 # (28 x 28)
num_classes=10 # MNSIST labels or classes
dropout=0.75 # Probability keep units


# In[5]:


# tf Graph input using placeholder ( valid only in TensorFlow version 1)

X=tf.placeholder(tf.float32,[None,num_input])
Y=tf.placeholder(tf.float32,[None,num_classes])
keep_prob=tf.placeholder(tf.float32)


# </div>
# <div class="alert alert-success" role="alert">
#   Step III: Define functions for image classifications such as <b><u>convolution, max pooling, full connected layer</u></b>
# </div>
# <p>
#   <button class="btn btn-primary" type="button" data-toggle="collapse" data-target="#multiCollapseExample2" aria-expanded="false" aria-controls="multiCollapseExample2">conv2d</button>
# </p>
#   <div class="col">
#     <div class="collapse multi-collapse" id="multiCollapseExample2">
#         <div class="card card-body">
#         <samp><mark>tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None, name=None
#               </mark></samp>
#           
#           input - A Tensor. 
#           
#           filters - A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width,
#           in_channels, out_channels]
#           
#           strides - An int or list of ints that has length 1, 2 or 4. The stride of the sliding window for each
#           dimension of input.
#           
#           padding - Either the string 'SAME' or 'VALID' indicating the type of padding algorithm to use

# <p>
#   <button class="btn btn-primary" type="button" data-toggle="collapse" data-target="#multiCollapseExample2" aria-expanded="false" aria-controls="multiCollapseExample2">maxpool2d</button>
# </p>
#   <div class="col">
#     <div class="collapse multi-collapse" id="multiCollapseExample2">
#         <div class="card card-body">
#           <samp><mark>tf.nn.max_pool(input, ksize, strides, padding, data_format=None, name=None)
#               </mark></samp>
# 
#           input - Tensor of rank N+2, of shape [batch_size] + input_spatial_shape + [num_channels]
#           
#           ksize - An int or list of ints that has length 1, N or N+2. The size of the window for each dimension of
#           the input tensor.
#           
#           strides - An int or list of ints that has length 1, N or N+2. The stride of the sliding window for each
#           dimension of the input tensor.
#               
#           padding - A string, either 'VALID' or 'SAME'.

# In[6]:


def conv2d(x,W,b,strides=1):
    x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x)


# In[7]:


def maxpool2d(x,k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')


# <h4><b><u>conv_total process</u></b></h4>
# 
# <span class="label label-pill label-primary">Reshape MNIST 1D to 4D [Batch Size, Height, Width, Channel]</span>
# 
# <span class="label label-pill label-info">First Convolution and Max Pooling Layer [5 x 5],1/32</span>
# 
# <span class="label label-pill label-success">Second Convolution and Max Pooling Layer [5 x 5],32/64</span>
# 
# <span class="label label-pill label-danger">Fully Connected Layer [7 x 7 x 64],1024</span>
# 
# <span class="label label-pill label-default">Dropout </span>
# 
# <span class="label label-pill label-warning">Output Layer [1024],10</span>
# 

# In[8]:


# Model Creation conv_total

def conv_total(x,weights,biases,dropout):
    
    x=tf.reshape(x,shape=[-1,28,28,1])
    
    conv1=conv2d(x,weights['wc1'],biases['bc1'])
    conv1=maxpool2d(conv1,k=2)
    
    conv2=conv2d(conv1,weights['wc2'],biases['bc2'])
    conv2=maxpool2d(conv2,k=2)
    
    fc1=tf.reshape(conv2,[-1,weights['wd1'].get_shape().as_list()[0]])
    fc1=tf.add(tf.matmul(fc1,weights['wd1']),biases['bd1'])
    fc1=tf.nn.relu(fc1)
    
    fc1=tf.nn.dropout(fc1,dropout)
    
    out=tf.add(tf.matmul(fc1,weights['out']),biases['out'])
    return out


# </div>
# <div class="alert alert-success" role="alert">
#   Step IV: <b><u>Weights, Biases</u></b> for all layers and output
# </div>

# In[9]:


weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}


# In[10]:


biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# </div>
# <div class="alert alert-success" role="alert">
#   Step V: Initialize and Define<b><u> Model, Loss, Optimizer, Evaluation</u></b>
# </div>

# In[11]:


#Model
model=conv_total(X,weights,biases,keep_prob)
pred=tf.nn.softmax(model)


# In[12]:


#Loss/Optimizer
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)


# In[13]:


# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# </div>
# <div class="alert alert-success" role="alert">
#   Step VI: Train<b><u> Model (Start Session)</u></b>
# </div>

# In[14]:


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if step % display_step == 0 or step == 1:
            loss_p, acc = sess.run([loss, accuracy],feed_dict={X: batch_x,Y: batch_y,keep_prob: 1.0})
            
            print("Step " + str(step) + ", Minibatch Loss= " +                   "{:.4f}".format(loss_p) + ", Training Accuracy= " +                   "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:",         sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
                                      keep_prob: 1.0}))    

