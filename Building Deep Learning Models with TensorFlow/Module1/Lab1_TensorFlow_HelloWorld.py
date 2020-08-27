# General introduction to working with TensorFlow
# TensorFlow defines computations as Graphs, and these are made with operations (also know as “ops”).
# So, when we work with TensorFlow, it is the same as defining a series of operations in a Graph. 

# Importing library
import tensorflow as tf

# Building a graph
graph1=tf.Graph() # tf.operation - node and tf.tensor - edge in a graph

# example - tf.constant add constants to the graph (operation). This operation produces the value and returns a tf.tensor that
# represents the value of the constant.

with graph1.as_default():
    a=tf.constant([2],name='constant_a')
    b=tf.constant([3],name='constant_b')

with open('Lab1_TensorFlow_HelloWorld.txt','a') as f:
    print('The tensor a is %s and b is %s' %(a,b),file=f)

# To print the values of a and b, we need to run the session for the graph
sess=tf.compat.v1.Session(graph=graph1)
result_a=sess.run(a)
result_b=sess.run(b)
with open('Lab1_TensorFlow_HelloWorld.txt','a') as f:
    print('\n',file=f)
    print('The value of a is %d and b is %d' %(result_a,result_b),file=f)


### Using an operation over the tensor a,b

with graph1.as_default():
    c=tf.add(a,b) # c=a+b

# Everytime, we need to run the session the print the values

result_c=sess.run(c)
with open('Lab1_TensorFlow_HelloWorld.txt','a') as f:
    print('\n',file=f)
    print('The value of c is %d' %(result_c),file=f)

#close session
sess.close()

# To avoid running and closing session everytime, we can write them within a block using 'with'

with tf.compat.v1.Session(graph=graph1) as sess2:
    result_a1=sess2.run(a)
    result_b1=sess2.run(b)
    result_c1=sess2.run(c)
    with open('Lab1_TensorFlow_HelloWorld.txt','a') as f:
        print('\n',file=f)
        print('The value a is %s \nThe value b is %s \nThe value c is %s' %(result_a1,result_b1,result_c1),file=f)

### Defining multi-dimesional array using TensorFlow ###

# Creating another graph as graph 2 so it doesn't coincide with graph1 in the same program
graph2=tf.Graph() # graph define
with graph2.as_default(): # adding constants
    Scalar=tf.constant(2) # for 0D - Single numbers
    Vector=tf.constant([5,6,2]) # for 1D - Series of numbers
    Matrix=tf.constant([[1,2,3],[2,3,4],[3,4,5]]) # for 2D - Table of Numbers
    Tensor=tf.constant([ [[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]] ]) # For 3D - cube of numbers
with tf.compat.v1.Session(graph=graph2) as sess_graph2:
    result_0=sess_graph2.run(Scalar)
    result_1=sess_graph2.run(Vector)
    result_2=sess_graph2.run(Matrix)
    result_3=sess_graph2.run(Tensor)
    with open('Lab1_TensorFlow_HelloWorld.txt','a') as f:
        print('\n',file=f)
        print('Scalar (0D) is: \n %s \nVector (1D) is: \n %s \nMatrix (2D) is: \n %s \nTensor (3D) is: \n %s' %(result_0,result_1,result_2,result_3),file=f)

# tf.shape returns the shape of data structure

with open('Lab1_TensorFlow_HelloWorld.txt','a') as f:
        print('\n',file=f)
        print('The shape of Scalar is %s and tensor is %s' %(Scalar.shape,Tensor.shape),file=f)

## Adding and multiplication multidimensional arrays
graph3= tf.Graph() # For addition
with graph3.as_default():
    M1=tf.constant([[1,2,3],[2,3,4],[3,4,5]]) # 2D matrix
    M2=tf.constant([[2,2,2],[2,2,2],[2,2,2]]) # 2D matrix

    add_1_tensor=tf.add(M1,M2)
    add_2_operation=M1+M2

graph4= tf.Graph() # For mulitplication
with graph4.as_default():
    M1=tf.constant([[2,3],[3,4]]) # 2D matrix
    M2=tf.constant([[2,3],[3,4]]) # 2D matrix

    mul_1_tensor=tf.matmul(M1,M2)
    mul_2_operation=M1*M2
    mul_3_dot=tf.tensordot(M1,M2,axes = 1)

# Printing addition and multiplication results
with tf.compat.v1.Session(graph=graph3) as sess_graph3:
    result_add_tensor=sess_graph3.run(add_1_tensor)
    result_add_ops=sess_graph3.run(add_2_operation)
    with open('Lab1_TensorFlow_HelloWorld.txt','a') as f:
        print('\n',file=f)
        print('The addition by tensorflow is : \n%s \nThe addition by general operation is : \n%s' %(result_add_tensor,result_add_ops),file=f)

with tf.compat.v1.Session(graph=graph4) as sess_graph4:
    result_mul_tensor=sess_graph4.run(mul_1_tensor)
    result_mul_ops=sess_graph4.run(mul_2_operation)
    result_mul_dot=sess_graph4.run(mul_3_dot)
    with open('Lab1_TensorFlow_HelloWorld.txt','a') as f:
        print('\n',file=f)
        print('The multiplication by tensorflow is : \n%s \nThe multiplication by general operation is : \n%s \nThe multiplication by dot operation: \n%s '%(result_mul_tensor,result_mul_ops,result_mul_dot),file=f)
