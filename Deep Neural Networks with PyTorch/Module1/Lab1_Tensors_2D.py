# Working on 2D Tensors basics -types, shape, indexing, slicing and operations

#Import Libraries
import numpy as np
import torch
import pandas as pd 
import matplotlib.pyplot as plt 

## Types and Shapes

twoD_list=[[11,12,13],[21,22,23],[31,32,33]]
twoD_tensor=torch.tensor(twoD_list)
with open('Lab1_Tensors_2D_Results.txt','a') as f:
    print('2D List <--> 2D Tensor: \n',file=f)
    print(twoD_tensor,file=f)
    print('\n',file=f)

with open('Lab1_Tensors_2D_Results.txt','a') as f:
    print('Basic elements: \n',file=f)
    print('The dimension of 2D tensor: ',twoD_tensor.ndimension(),file=f)
    print('The shape of 2D tensor: ',twoD_tensor.shape,file=f)
    print('The size of 2D tensor: ',twoD_tensor.size(),file=f)
    print('The no. of elements of 2D tensor: ',twoD_tensor.numel(),file=f)
    print('\n',file=f)

## Numpy <--> Tensor, Pandas DF --> Tensor

twoD_numpy=twoD_tensor.numpy()
new_twoD_tensor=torch.from_numpy(twoD_numpy)

with open('Lab1_Tensors_2D_Results.txt','a') as f:
    print('Numpy --> Tensor: \n',file=f)
    print('The numpy after converting:\n',twoD_numpy,file=f)
    print('The type of numpy: ',twoD_numpy.dtype,file=f)
    print('-------------------------------------------',file=f)
    print('The tensor from numpy:\n',new_twoD_tensor,file=f)
    print('The type of new tensor from numpy: ',new_twoD_tensor.dtype,file=f)
    print('\n',file=f)

df=pd.DataFrame({'a':[11,21,31],'b':[12,22,32]})
new_df_tensor=torch.from_numpy(df.values)
with open('Lab1_Tensors_2D_Results.txt','a') as f:
    print('Pandas DF --> Tensor: \n',file=f)
    print('Pandas to numpy:\n',df.values,file=f)
    print('Type before conversion: ',df.values.dtype,file=f)
    print('-------------------------------------------',file=f)
    print('The tensor after converting:\n',new_df_tensor,file=f)
    print('The type of new tensor after converting: ',new_df_tensor.dtype,file=f)
    print('\n',file=f)

## Indexing and Slicing
tensor_example=torch.tensor(twoD_list)
with open('Lab1_Tensors_2D_Results.txt','a') as f:
    print('Indexing: \n',file=f)
    print('Value on 2nd row and 3rd column: ',tensor_example[1,2],file=f)
    print('Value on 2nd row and 3rd column: ',tensor_example[1][2],file=f)
    print('-------------------------------------------',file=f)
    print('Indexing and Slicing: \n',file=f)
    print('Value on 1st row and first two columns: ',tensor_example[0,0:2],file=f)
    print('Value on 1st row and first two columns: ',tensor_example[0][0:2],file=f)
    print('\n',file=f)

## Operations - Add, Scalar Multiplication, Element-wise Product, Matrix Mulitplication

X=torch.tensor([[1,0],[0,1]])
Y=torch.tensor([[2,1],[1,2]])

XYadd=X+Y # Addition
Z=2*Y #Multilpcation
XY_times=X*Y #Element wise product

#If X and Y are not same dimension matrix then X*Y is not necessarily same as Y*X
X_new=torch.tensor([[0,1,1],[1,0,1]]) # 2 x 3 matrix
Y_new=torch.tensor([[1,1],[1,1],[-1,1]]) # 3 x 2 matrix
XY_mm=torch.mm(X_new,Y_new) # 2 x 2 result of product

with open('Lab1_Tensors_2D_Results.txt','a') as f:
    print('Operations: \n',file=f)
    print('Adding X and Y gives:\n',XYadd,file=f)
    print('Scalar Mutiplication:\n',Z,file=f)
    print('Element Wise Product:\n',XY_times,file=f)
    print('Matrix Multiplication Xnew,Ynew:\n',XY_mm,file=f)
    print('\n',file=f)

    
