# In 2 part for 1D Linear Regression, we will train the modeil with PyTorch

import numpy as np
import matplotlib.pyplot as plt
import torch

# define class plot_diagram - to visualize data space and parameter space during training

class plot_diagram():

    def __init__(self,X,Y,w,stop,go=False): #Constructor
        start = w.data

        self.error=[]
        self.parameter=[]
        self.X=X.numpy()
        self.Y=Y.numpy()
        
        self.parameter_values=torch.arange(start,stop)
        self.Loss_function=[criterion(forward(X),Y) for w.data in self.parameter_values]
        w.data=start
    
    def __call__(self, Yhat,w,error,n): #Executor
        self.error.append(error)
        self.parameter.append(w.data)

        plt.subplot(212)
        plt.plot(self.X,Yhat.detach().numpy())
        plt.plot(self.X,self.Y,'ro')

        plt.xlabel('A')
        plt.ylim(-20,20)
        
        plt.subplot(211)
        plt.title('Data Space (Top) Estimated Line (Bottom) Iteration' + str(n))
        plt.plot(self.parameter_values.numpy(),self.Loss_function)
        plt.plot(self.parameter,self.error,'ro')
        plt.xlabel('B')
        plt.figure()

    def __del__(self): #Destructor
        plt.close('all')

## Make some Data - generate values -3 to 3 that create a line with a slope of 3

X=torch.arange(-3,3,0.1).view(-1,1)
f=-3*X
# Add some noise
Y=f+0.1*torch.randn(X.size())

plt.figure() #plotting line and added noise
plt.plot(X.numpy(),Y.numpy(),'rx',label='Y')
plt.plot(X.numpy(),f.numpy(),label='f')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Model and cost function built

def forward(x): #for prediction
    return w*x
def forward2(x): #for prediction
    return w2*x

def criterion(yhat,y): # for MSE to evaluate the result
    return torch.mean((yhat - y) ** 2)

lr=0.1
LOSS=[] #loss matrix 1
LOSS2=[] #loss matrix 2

w=torch.tensor(-10.0,requires_grad=True) #Learnable paramter 1
w2=torch.tensor(-15.0,requires_grad=True) #Learnable paramter 2

gradient_plot=plot_diagram(X,Y,w,stop=5)
gradient_plot2=plot_diagram(X,Y,w2,stop=15)

# Train model

#define a function to train a model:

def train_model(iter): # for w and LOSS
    for epoch in range(iter):

        Yhat=forward(X) #make prediction
        loss=criterion(Yhat,Y) # calculate iteration
        gradient_plot(Yhat,w,loss.item(),epoch) #plot diagram

        LOSS.append(loss.item()) #store loss into list

        loss.backward() # compute gradient of loss with wrt to all learnable parameters
        w.data=w.data-lr*w.grad.data #update parameters
        w.grad.data.zero_() # xero the gradient before backward pass

def my_train_model(iter): # for w2 and LOSS2
    for epoch in range(iter):

        Yhat2=forward2(X) #make prediction
        loss2=criterion(Yhat2,Y) # calculate iteration
        gradient_plot2(Yhat2,w2,loss2.item(),epoch) #plot diagram

        LOSS2.append(loss2.item()) #store loss into list

        loss2.backward() # compute gradient of loss with wrt to all learnable parameters
        w2.data=w2.data-lr*w2.grad.data #update parameters
        w2.grad.data.zero_() # xero the gradient before backward pass


train_model(4) # model train with plot for 4 iterations
my_train_model(4)

# Plot loss for each iteration

plt.figure()
plt.plot(LOSS)
plt.plot(LOSS2)
plt.tight_layout()
plt.xlabel('Epoch/Iterations')
plt.ylabel('Cost')

#Disply plots
plt.show()

