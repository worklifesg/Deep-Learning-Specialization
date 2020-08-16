# Obviously, neural networks for real problems are composed of many hidden layers 
# and many more nodes in each layer. So, we can't continue making predictions using 
# this very inefficient approach of computing the weighted sum at each node 
# and the activation of each node manually.

# In order to code an automatic way of making predictions, let's generalize our network.
# A general network would take  𝑛  inputs, would have many hidden layers, each hidden 
# layer having  𝑚  nodes, and would have an output layer. Although the network is 
# showing one hidden layer, but we will code the network to have many hidden layers. 
# Similarly, although the network shows an output layer with one node, we will code 
# the network to have more than one node in the output layer.

# Initialize network

n=2 #inputs
n_hid=2 #hidden layers
m=[2,2] #nodes in each hidden layer
n_out=1 #nodes in output layer

import numpy as np
n_nodes_p=n # number of nodes in the previous layer

# intialize network
network={}

# Creating loops for each layer with their weights and biases associated with each node
# To include the output layer, add 1 to number of hidden layers

for layer in range(n_hid+1):
    #----------- layer determination -------------
    if layer == n_hid:
        layer_name='output' # to define last layer as output layer
        n_nodes=n_out
    else:
        layer_name ='layer_{}'.format(layer+1) # to name each layer as layer 1, layer 2....
        n_nodes=m[layer]
    #------------ weights and biases ---------------
    network[layer_name]={}
    for node in range(n_nodes):
        node_name='node_{}'.format(node+1) # to define nodes for each layer
        network[layer_name][node_name]={
            'weights': np.around(np.random.uniform(size=n_nodes_p),decimals=2),
            'bias': np.around(np.random.uniform(size=1),decimals=2) # define weights and bias for each node in each layer
        }
    n_nodes_p=n_nodes
with open ("Lab1_ANN_Networks.txt",'a') as f:
    print(network,file=f)
        
## For the above code, let us create a function that n (inputs),n_hid (hidden layers), m (nodes in each hidden layer)
## and n_out(nodes in output layer)

def intialize_network(n,n_hid,m,n_out):
    n_nodes_p=n 
    network={}
    for layer in range(n_hid+1):
            if layer == n_hid:
                layer_name='output' # to define last layer as output layer
                n_nodes=n_out
            else:
                layer_name ='layer_{}'.format(layer+1) # to name each layer as layer 1, layer 2....
                n_nodes=m[layer]
            network[layer_name]={}
            for node in range(n_nodes):
                 node_name='node_{}'.format(node+1) # to define nodes for each layer
                 network[layer_name][node_name]={
                     'weights': np.around(np.random.uniform(size=n_nodes_p),decimals=2),
                     'bias': np.around(np.random.uniform(size=1),decimals=2) # define weights and bias for each node in each layer
                     }
            n_nodes_p=n_nodes
    return network

#--------- Exercise 1 = create a small network-----------------

ns=5 #inputs
n_hids=3 #hidden layers
ms=[3,2,3] #nodes in each hidden layer
n_outs=1 #nodes in output layer

small_network=intialize_network(ns,n_hids,ms,n_outs)
with open ("Lab1_ANN_Networks.txt",'a') as f:
    print('Small network is: \n',small_network,file=f)
