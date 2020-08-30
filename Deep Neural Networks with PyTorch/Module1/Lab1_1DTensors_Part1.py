# Fundamental of Tensor Operations - 1D

#Import libraries
import torch
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

# Plot vecotrs, please keep the parameters in the same length
# @param: Vectors = [{"vector": vector variable, "name": name of vector, "color": color of the vector on diagram}]

def plotVec(vectors):
    ax=plt.axes() #defining axis

    #loop to draw the vector
    for vec in vectors:
        ax.arrow(0,0,*vec['vector'],head_width=0.05,
        color=vec['color'],head_length=0.1)
        plt.text(*(vec['vector']+0.1),vec['name'])
    plt.ylim(-2,2)
    plt.xlim(-2,2)

## Type (.type()) and shape (.dtype) of tensor

#------------Convert a integer list with length 5 to a tensor
ints_to_tensor=torch.tensor([0,1,2,3,4])
with open('Lab1_Tensors_1D_Results.txt','a') as f:
    print('Integer to tensor: \n',file=f)
    print('Dtype of tensor after converting it to tensor',ints_to_tensor.dtype,file=f)
    print('Type of tensor after converting it to tensor',ints_to_tensor.type(),file=f)
    print('\n',file=f)
# the integer list has been converted to a long tensor. But python type is still torch

#------------Convert a float list with length 5 to a tensor
floats_to_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
with open('Lab1_Tensors_1D_Results.txt','a') as f:
    print('Float to tensor: \n',file=f)
    print("The dtype of tensor object after converting it to tensor: ", floats_to_tensor.dtype,file=f)
    print("The type of tensor object after converting it to tensor: ", floats_to_tensor.type(),file=f)
    print('\n',file=f)

#-------------Convert float to integer
list_floats=[0.0,1.0,2.0,3.0,4.0]
floats_int_tensor=torch.tensor(list_floats,dtype=torch.int64)
with open('Lab1_Tensors_1D_Results.txt','a') as f:
    print('Float to Integer: \n',file=f)
    print("The dtype of tensor is : ", floats_int_tensor.dtype,file=f)
    print("The type of tensor is : ", floats_int_tensor.type(),file=f)
    print('\n',file=f)

# Alternatively, using specific methods in torch, conversions can be implemented as well.

new_float_tensor=torch.FloatTensor([0,1,2,3,4])
with open('Lab1_Tensors_1D_Results.txt','a') as f:
    print('Integer to float(method): \n',file=f)
    print("The type of new_float_tensor is : ", new_float_tensor.type(),file=f)
    print('\n',file=f)

# You can also convert an existing tensor object (tensor_obj) to another tensor type

#---------------Convert the integer tensor to a float tensor:
old_int_tensor=torch.tensor([0,1,2,3,4])
new_float_tensor1=old_int_tensor.type(torch.FloatTensor)
with open('Lab1_Tensors_1D_Results.txt','a') as f:
    print('Integer to float(Another method): \n',file=f)
    print("The type of new_float_tensor1 is : ", new_float_tensor1.type(),file=f)
    print('\n',file=f)

### Tensor Object  - Size (.size()), Dimension (.ndimension()), Reshaping (.view(row,column))

# Reshaping
twoD_float_tensor = new_float_tensor.view(5, 1) # reshaping 1D to 2D matrix
with open('Lab1_Tensors_1D_Results.txt','a') as f:
    print('Size,Dimension,Reshaping \n',file=f)
    print("The size of new_float_tensor1 is : ", new_float_tensor1.size(),file=f)
    print("The dimension of new_float_tensor1 is : ", new_float_tensor1.ndimension(),file=f) #Dimension =1
    print("The original size new_float_tensor1 is : ", new_float_tensor1, file=f)
    print("The modified of new_float_tensor1 is : \n ", twoD_float_tensor,file=f)
    print('\n',file=f)

# If we have a tensor with dynamic size and wants to reshape it.
twoD_float_tensor1 = new_float_tensor.view(-1, 1)
with open('Lab1_Tensors_1D_Results.txt','a') as f:
    print('Reshaping using general and dynamic form (-1) \n',file=f)
    print("The original size new_float_tensor1 is : ", new_float_tensor1, file=f)
    print("The modified of new_float_tensor1 is : \n ", twoD_float_tensor,file=f)
    print("The modified of new_float_tensor1 using  is : \n ", twoD_float_tensor1,file=f)
    print("The size of twoD_float_tensor1 is : ", twoD_float_tensor1.size(),file=f)
    print("The dimension of twoD_float_tensor1 is : ", twoD_float_tensor1.ndimension(),file=f) #Dimension=2
    print('\n',file=f)

#----------------- Convert numpy array and pandas series to a tensor or vice-versa------------------

# -using torch.from_numpy(), <file_tensor_name>.numpy()

np_array=np.array([0.0,1.0,2.0,3.0,4.0])
new_tensor=torch.from_numpy(np_array)

back_to_numpy=new_tensor.numpy()

with open('Lab1_Tensors_1D_Results.txt','a') as f:
    print('Numpy <--> Tensor: \n',file=f)
    print("The dtype of new tensor is : ", new_tensor.dtype,file=f)
    print("The type of new tensor is : ", new_tensor.type(),file=f)
    print("The dtype of numpy from tensor is : ", back_to_numpy.dtype,file=f)
    print("The numpy array from tensor is : ", back_to_numpy,file=f)   
    print('\n',file=f)

#------ Please note: back_to_numpy and new_tensor still point to numpy_array. 
# As a result if we change numpy_array both back_to_numpy and new_tensor will change.

# Pandas Series can also be converted by using the numpy array that is stored in pandas_series.values.

pd_series=pd.Series([0.1,0.2,0.3,10.1])
new_tensor_pd=torch.from_numpy(pd_series.values)

with open('Lab1_Tensors_1D_Results.txt','a') as f:
    print('Pandas <--> Tensor: \n',file=f)
    print("The new tensor from numpy array: ", new_tensor_pd,file=f)
    print("The dtype of new tensor: ", new_tensor_pd.dtype,file=f)
    print("The type of new tensor: ", new_tensor_pd.type(),file=f)  
    print('\n',file=f)

#Using .item() - returns of value of tensor as number, tolist() - to return a list

this_tensor=torch.tensor([0,1,2,3])
torch_to_list=this_tensor.tolist()
with open('Lab1_Tensors_1D_Results.txt','a') as f:
    print('Illustration of .item and .tolist: \n',file=f)
    print("The first item is given by: ",this_tensor[0].item(),' The first rensor value is given by: ',this_tensor[0],file=f)
    print("The first item is given by: ",this_tensor[1].item(),' The first rensor value is given by: ',this_tensor[1],file=f)
    print("The first item is given by: ",this_tensor[2].item(),' The first rensor value is given by: ',this_tensor[2],file=f)
    print("Tensor: ", this_tensor,'\nlist:',torch_to_list, file=f)
