# In 3 part for 1D Linear Regression, we will train the model with PyTorch with Slope and Bias

import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits import mplot3d

# define class plot_diagram - to visualize data space and parameter space during training

class plot_diagram(object):

    def __init__(self,w_range,b_range,X,Y,n_samples=30,go=True): #Constructor
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z = np.zeros((30,30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize = (7.5, 5))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride = 1, cstride = 1,cmap = 'viridis', edgecolor = 'none')
            plt.title('Cost/Total Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Cost/Total Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
    
    # Setter
    def set_para_loss(self, W, B, loss):
        self.n = self.n + 1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)
    
    # Plot diagram
    def final_plot(self): 
        ax = plt.axes(projection = '3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W,self.B, self.LOSS, c = 'r', marker = 'x', s = 200, alpha = 1)
        plt.figure()
        plt.contour(self.w,self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
    
    # Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x, self.y, 'ro', label="training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label = "estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-10, 15))
        plt.title('Data Space Iteration: ' + str(self.n))

        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.title('Total Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()

## Make some Data - generate values -3 to 3 that create a line with a slope of 3

X=torch.arange(-3,3,0.1).view(-1,1)
f=1*X-1
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
    return w*x+b

def criterion(yhat,y): # for MSE to evaluate the result
    return torch.mean((yhat - y) ** 2)

lr=0.1
lr2=0.2
LOSS=[] #loss matrix 1
LOSS2=[] #loss matrix 2

w=torch.tensor(-15.0,requires_grad=True) #Learnable paramter 1
b=torch.tensor(-10.0,requires_grad=True) #Learnable paramter 2

get_plot=plot_diagram(15,15,X,Y,30)


# Train model

#define a function to train a model:

def train_model(iter): # for w and LOSS
    for epoch in range(iter):

        Yhat=forward(X) #make prediction
        loss=criterion(Yhat,Y) # calculate iteration
        get_plot.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
        if epoch % 3 == 0:
            get_plot.plot_ps()

        LOSS.append(loss) #store loss into list

        loss.backward() # compute gradient of loss with wrt to all learnable parameters

        w.data=w.data-lr*w.grad.data #update parameters
        b.data=b.data-lr*b.grad.data

        w.grad.data.zero_() # xero the gradient before backward pass
        b.grad.data.zero_()

def my_train_model(iter): # for w2 and LOSS2
    for epoch in range(iter):

        Yhat=forward(X) #make prediction
        loss=criterion(Yhat,Y) # calculate iteration
        get_plot.set_para_loss(w.data.tolist(), b.data.tolist(), loss.tolist())
        if epoch % 3 == 0:
            get_plot.plot_ps()

        LOSS2.append(loss) #store loss into list

        loss.backward() # compute gradient of loss with wrt to all learnable parameters

        w.data=w.data-lr2*w.grad.data #update parameters
        b.data = b.data - lr2 * b.grad.data
        
        w.grad.data.zero_() # xero the gradient before backward pass
        b.grad.data.zero_()


train_model(15) # model train with plot for 4 iterations
my_train_model(15)

# Plot loss for each iteration

plt.figure()
plt.plot(LOSS)
plt.plot(LOSS2)
plt.tight_layout()
plt.xlabel('Epoch/Iterations')
plt.ylabel('Cost')

#Disply plots
plt.show()
