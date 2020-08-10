# Neural network has 2 inputs, 1 hidden layer with 2 nodes and 1 output layer with one node.
# x1,x2 - I/P, w1,w2,w3,w4 - I/P weights, b1,1 b1,2 - biases, w5,w6 - Hidden layer weights
# b2 - bias, a2 - O/P, z1,1 z1,2 z2 - linear combination of weights and biases

#Initializing weights and biases
import numpy as np
weights=np.around(np.random.uniform(size=6),decimals=2)
biases=np.around(np.random.uniform(size=3),decimals=2)

# Given x1 x2
x1=0.5
x2=0.85

with open ("Lab1_ANN.txt",'a') as f:
    print("The weights are: ", weights,file=f)
    print("The biases are: ", biases,file=f)
    print("x1 is {} and x2 is {}".format(x1,x2),file=f)

# Computing z1,1 z1,2

z11=x1*weights[0]+x2*weights[1]+biases[0]
z12=x1*weights[2]+x2*weights[3]+biases[1]

with open ("Lab1_ANN.txt",'a') as f:
    print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z11, decimals=3),file=f)
    print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(z12, decimals=3),file=f)

#using sigmoid function to compute a1,1 a1,2

a11=1.0/(1.0+np.exp(-z11))
a12=1.0/(1.0+np.exp(-z12))

with open ("Lab1_ANN.txt",'a') as f:
    print('The activation of the first node in the hidden layer is {}'.format(a11, decimals=3),file=f)
    print('The activation of the second node in the hidden layer is {}'.format(a12, decimals=3),file=f)

# computing z2 for O/P layer using hidden layer inputs a1,1 a1,2
z2=a11*weights[4]+a12*weights[5]+biases[2]

# Computing O/P layer term a2
a2=1.0/(1.0+np.exp(-z2))

with open ("Lab1_ANN.txt",'a') as f:
    print('The weighted sum of the inputs at the node in the output layer is {}'.format(z2, decimals=3),file=f)
    print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(a2, decimals=3),file=f)
