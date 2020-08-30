# Fundamental concepts of Tensors - Index/Slicing and Functions

#Import libraries
import torch
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

#---------------------Indexing and Slicing ---------------------------

#indexing on tensors
index_tensor=torch.tensor([0,1,2,3,4])
with open('Lab1_Tensors_1D_Results_2.txt','a') as f:
    print('Indexing: \n',file=f)
    print('The value on index 0: ',index_tensor[0],file=f)
    print('The value on index 1: ',index_tensor[1],file=f)
    print('The value on index 2: ',index_tensor[2],file=f)
    print('The value on index 3: ',index_tensor[3],file=f)
    print('The value on index 4: ',index_tensor[4],file=f) #starts from 0th index and always (n-1) indices where n is total number of elements
    print('\n',file=f)

#Change the value of certain index
tensor_sample=torch.tensor([20,1,2,3,4]) #defining tensor
with open('Lab1_Tensors_1D_Results_2.txt','a') as f:
    print('Indexing Change: \n',file=f)
    print('The value on index 0: ',tensor_sample[0],file=f)

tensor_sample[0]=100 #changing 0th index from 20 to 100
with open('Lab1_Tensors_1D_Results_2.txt','a') as f:
    print('The modified tensor: ',tensor_sample,file=f)
    print('\n',file=f)

with open('Lab1_Tensors_1D_Results_2.txt','a') as f:
    print('Indexing Change: \n',file=f)
    print('The value on index 4: ',tensor_sample[4],file=f)
tensor_sample[4]=0 #changing 4th index to 0
with open('Lab1_Tensors_1D_Results_2.txt','a') as f:
    print('The modified tensor: ',tensor_sample,file=f)
    print('\n',file=f)

#Slicing [i:n-1], example [1:4] will return only index 1, 2, 3 and not 4.

subset_tensor_sample=tensor_sample[1:4]
with open('Lab1_Tensors_1D_Results_2.txt','a') as f:
    print('Slicing: \n',file=f)
    print('Original Tensor: ',tensor_sample,file=f)
    print('Modified Tensor: ',subset_tensor_sample,file=f)
    print('\n',file=f)

#Assigning certain index some values to the slices
with open('Lab1_Tensors_1D_Results_2.txt','a') as f:
    print('Indexing and Slicing: \n',file=f)
    print('Intial value of index 3 and 4: ',tensor_sample[3:5],file=f)
tensor_sample[3:5]=torch.tensor([300.0,400.0])
with open('Lab1_Tensors_1D_Results_2.txt','a') as f:
    print('Modified Tensor: ',tensor_sample,file=f)
    print('\n',file=f)

#------------------------Functions-------------------------------------

#Mean, Standard Deviation, Max, Min, Sin,linspace

math_tensor=torch.tensor([1.0, 1.0, 3.0, 5.0, 5.0])

mean=math_tensor.mean()
std=math_tensor.std()
max_val=math_tensor.max()
min_val=math_tensor.min()

#Max/Min gives the maximum or minimum value but not the elements that contain the maximum or minimum value in the tensor.

pi_tensor=torch.tensor([0,np.pi/2,np.pi])
sin_tensor=torch.sin(pi_tensor)

with open('Lab1_Tensors_1D_Results_2.txt','a') as f:
    print('Functions: \n',file=f)
    print('Mean: ',mean,file=f)
    print('Standard Deviation: ',std,file=f)
    print('Max: ',max_val,file=f)
    print('Min: ',min_val,file=f)
    print('Result of pi tensor (sin): ',sin_tensor,file=f)
    print('\n',file=f)

#Plotting sin graph using linspace

pi_tensor1=torch.linspace(0,2*np.pi,100)
sin_result=torch.sin(pi_tensor1)

plt.plot(pi_tensor1.numpy(),sin_result.numpy())
plt.show()
