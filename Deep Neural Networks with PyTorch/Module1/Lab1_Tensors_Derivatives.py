# Derivatives and Partial Derivatives using PyTorch

#Import Libraries
import torch
import matplotlib.pyplot as plt 


x=torch.tensor(2.0,requires_grad=True) #Create a tensor x
y=x**2 # Create a tensor y according to y = x^2
y.backward() # Take the derivative

with open('Lab1_Tensors_Derivatives_Results.txt','a') as f:
    print('Derivative Results: \n',file=f)
    print('The tensor x: ',x,file=f)
    print('The result of y=x^2: ',y,file=f)
    print('The derivative at the value x = 2: ',x.grad,file=f) # to print out the derivative at the value x = 2
    print('\n',file=f)

# Printing the attributes of both tensor (x) and derviative function (y)

with open('Lab1_Tensors_Derivatives_Results.txt','a') as f:
    print('Tensor x Attributes and Values: \n',file=f)
    print('data:',x.data,file=f) # Data of tensor
    print('grad_fn:',x.grad_fn,file=f) # points to the node in backward graph
    print('grad:',x.grad,file=f) #gradient or derivative value once it is calculated
    print("is_leaf:",x.is_leaf,file=f) #whether a particular node is leaf in the graph or not
    print("requires_grad:",x.requires_grad,file=f) # True if derivative is to be calculated at specified value of x
    print('-----------------------------------------------',file=f)
    print('Derivatives of y Attributes and Values: \n',file=f)
    print('data:',y.data,file=f)
    print('grad_fn:',y.grad_fn,file=f)
    print('grad:',y.grad,file=f)
    print("is_leaf:",y.is_leaf,file=f)
    print("requires_grad:",y.requires_grad,file=f)
    print('\n',file=f)

## Own Custom Autograd Functions
# A staticmethod is a method that knows nothing about the class or instance it was called on. 
# It just gets the arguments that were passed, no implicit first argument. 
"""
In the forward pass we receive a Tensor containing the input and return
a Tensor containing the output. ctx is a context object that can be used
to stash information for backward computation. You can cache arbitrary
objects for use in the backward pass using the ctx.save_for_backward method.

In the backward pass we receive a Tensor containing the gradient of the loss
with respect to the output, and we need to compute the gradient of the loss
with respect to the input.

"""
class SQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx,i):
        result=i**2
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx,grad_output):
        i,=ctx.saved_tensors
        grad_output=2*i
        return grad_output

## Calling SQ class to evlaute the derivative
x_1=torch.tensor(2.0,requires_grad=True)
sq=SQ.apply

y=sq(x_1)
y.backward()
with open('Lab1_Tensors_Derivatives_Results.txt','a') as f:
    print('Derivative Results using SQ Class: \n',file=f)
    print('The derivative at the value x = 2: ',x_1.grad,file=f) # to print out the derivative at the value x = 2
    print('\n',file=f)

## Partial Derivatives
u =torch.tensor(1.0,requires_grad=True)
v =torch.tensor(2.0,requires_grad=True)
f=u*v+u**2

f.backward()
with open('Lab1_Tensors_Derivatives_Results.txt','a') as f:
    print('Partial Derivative Results: \n',file=f)
    print('The partial derivative wrt u: ',u.grad,file=f) 
    print('The partial derivative wrt v: ',v.grad,file=f) 
    print('\n',file=f)

## Derivatives for multiple values
plt.figure()
xm=torch.linspace(-10,10,10,requires_grad=True)
ym=xm**2
ym1=torch.sum(xm**2)
ym1.backward()
plt.plot(xm.detach().numpy(), ym.detach().numpy(), label = 'function')
plt.plot(xm.detach().numpy(), xm.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()


## Derivative using mutiple values for 'ReLu' activation function
plt.figure()
xre = torch.linspace(-10, 10, 1000, requires_grad = True)
Yre = torch.relu(xre)
yre = Yre.sum()
yre.backward()
plt.plot(xre.detach().numpy(), Yre.detach().numpy(), label = 'function')
plt.plot(xre.detach().numpy(), xre.grad.detach().numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()
