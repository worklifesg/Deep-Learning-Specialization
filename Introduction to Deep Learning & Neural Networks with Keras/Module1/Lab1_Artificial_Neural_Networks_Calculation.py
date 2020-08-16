# Here we will compute the Weighted Sum at Each Node (z) and Node Activation (a-sigmoid function)
# But will define small_network from Lab1_Artificial_Neural_Networks_Networks

import numpy as np

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

ns=5 #inputs
n_hids=3 #hidden layers
ms=[3,2,3] #nodes in each hidden layer
n_outs=1 #nodes in output layer

small_network=intialize_network(ns,n_hids,ms,n_outs)

## Computing weighted sum at each Node - we need inputs ,weights, biases

def compute_weighted_sum(inputs,weights,bias):
    return np.sum(inputs*weights)+bias

# Example print the weighted sum at layer 1 node 1 (first node at first hidden layer)

np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2) # random inputs

node_weights=small_network['layer_1']['node_1']['weights']
node_bias=small_network['layer_1']['node_1']['bias']

weighted_sum=compute_weighted_sum(inputs,node_weights,node_bias)

with open ("Lab1_ANN_Compute.txt",'a') as f:
    print('Inputs are: \n',inputs,file=f)
    print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)),file=f)


## Node activation (Sigmoid function)

def node_activation(weighted_sum):
    return 1.0/(1.0+np.exp(-1*weighted_sum))

# Example print the node activation at layer 1 node 1 (first node at first hidden layer)

node_a_output=node_activation(compute_weighted_sum(inputs,node_weights,node_bias))
with open ("Lab1_ANN_Compute.txt",'a') as f:
    print('The output at the first node in the hidden layer is {}'.format(np.around(node_a_output[0], decimals=4)),file=f)
