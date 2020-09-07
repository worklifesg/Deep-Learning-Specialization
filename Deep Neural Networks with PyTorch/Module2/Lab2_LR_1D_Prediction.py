# in this lab we would review how to have preidctions using Pytorch
# -- prediction, class linear , build cusotm modules

#--- Prediction
import torch
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt 

# y= b+wx b- biases and w -weights

w=torch.tensor(2.0,requires_grad=True)
b=torch.tensor(-1.0,requires_grad=True)

# to function make a prediction based on weight and biases
def forward(x):
    yhat=w*x+b
    return yhat

x=torch.tensor([[1.0]])
yhat=forward(x)
with open ('Lab2_Linear_Regression_1D_Prediction_Results.txt','a') as f:
    print('The prediction is', yhat,file=f)

# for multiple inputs
x_1=torch.tensor([[1.0],[2.0]])
with open ('Lab2_Linear_Regression_1D_Prediction_Results.txt','a') as f:
    print('the shape of x1 is: ', x_1.shape,file=f)

yhat_1=forward(x_1)
with open ('Lab2_Linear_Regression_1D_Prediction_Results.txt','a') as f:
    print('The prediction for multiple input values is: ', yhat_1,file=f)

#-------------------------------------------------------------------------------#

# class linear - used to make predictions

from torch.nn import Linear

torch.manual_seed(1)

lr=Linear(in_features=1,out_features=1,bias=True)
with open ('Lab2_Linear_Regression_1D_Prediction_Results.txt','a') as f:
    print('The parameters w and b are:\n', list(lr.parameters()),file=f)

#state_dict() - keys (weight and bias) and values (their corresponding values)
with open ('Lab2_Linear_Regression_1D_Prediction_Results.txt','a') as f:
    print('My Python dictionary: ',lr.state_dict(),file=f)
    print('Keys: ', lr.state_dict().keys(),file=f)
    print('Values: ', lr.state_dict().values(),file=f)
    print('Weight: ',lr.weight,file=f)
    print('Bias: ',lr.bias,file=f)

# using the class linear to predict values of single input and multiple values
x_linear=torch.tensor([[1.0]])
x_linear_mult=torch.tensor([[1.0],[2.0]])
yhat_linear=lr(x_linear)
yhat_linear_mult=lr(x_linear_mult)
with open ('Lab2_Linear_Regression_1D_Prediction_Results.txt','a') as f:
    print('The prediction using class linear is: ',yhat_linear,file=f)
    print('The prediction of multiple input values using class linear: \n', yhat_linear_mult,file=f)

# Build custom modules

from torch import nn
# Customize Linear Regression Class

class LR(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        
        # Inherit from parent
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
    def forward(self, x):
        out = self.linear(x)
        return out

# Create the linear regression model. Print out the parameters.

lr_class = LR(1, 1)
with open ('Lab2_Linear_Regression_1D_Prediction_Results.txt','a') as f:
    print("The parameters using custom module: ", list(lr_class.parameters()),file=f)
    print("Linear model using custom module: ", lr_class.linear,file=f)

# Try our customize linear regression model with single input

x_class = torch.tensor([[1.0]])
yhat_class = lr_class(x_class)
x_class_mult = torch.tensor([[1.0], [2.0]])
yhat_class_mult = lr_class(x_class_mult)
with open ('Lab2_Linear_Regression_1D_Prediction_Results.txt','a') as f:
    print("The prediction using custom module for single value: ", yhat_class,file=f)
    print("The prediction using custom module for multiple value: ", yhat_class_mult,file=f)
    print("Python dictionary: ", lr_class.state_dict(),file=f)
    print("keys: ",lr_class.state_dict().keys(),file=f)
    print("values: ",lr_class.state_dict().values(),file=f)
