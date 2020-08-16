## For final code, we have 3 functions - intialize_network, compute_weighted_sum and node_activation.
# We combine compute_weighted_sum and node_activation function to evaluate predictions for each node at every hiiden layer and final output prediction
# Combine function is called forward_propagate
# So we intialize network, define input and find predictions using forward_propagate function

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
    with open ("Lab1_ANN_Final.txt",'a') as f:
        print('The network is: \n',network,file=f)
    return network

def compute_weighted_sum(inputs,weights,bias):
    return np.sum(inputs*weights)+bias

def node_activation(weighted_sum):
    return 1.0/(1.0+np.exp(-1*weighted_sum))

def forward_propagate(network,inputs):
    layer_inputs=list(inputs) 

    for layer in network:
        layer_data=network[layer]
        layer_outputs=[]

        for layer_node in layer_data:
            node_data=layer_data[layer_node]

            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
        
        if layer !='output':
            with open ("Lab1_ANN_Final.txt",'a') as f:
                print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs),file=f)
        
        layer_inputs = layer_outputs

    network_predictions=layer_outputs
    return network_predictions

## Program ##

ns=5 #inputs
n_hids=3 #hidden layers
ms=[2,3,2] #nodes in each hidden layer
n_outs=3 #nodes in output layer

small_network=intialize_network(ns,n_hids,ms,n_outs)

np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2) # random inputs

predictions = forward_propagate(small_network, inputs)

with open ("Lab1_ANN_Final.txt",'a') as f:
    print('Inputs are: \n',inputs,file=f)
    print('The predicted value by the network for the given input is {}'.format(predictions),file=f)
