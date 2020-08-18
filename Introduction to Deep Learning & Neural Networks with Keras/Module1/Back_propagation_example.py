# Program to illustrate simple Backpropagation example #

# Step 1. Use simple forward propagation technique using intial values of w1,w2 and b1,b2 to find z1,z2,a,a2
# Step 2. Assume Ground truth T different than predicted output
# Step 3. Calculate MSE - E with respect to a2 (output) for propagating back into the network.
# Step 4. Update w2,b2,w1,b1
# Step 5. Repeat number of steps till E --> 0

# Assumptions: number of iterations = 1000, predefined threshold = 0.001, Learning Rate - 0.4.

## Forward propagation to calculate z1,a1,z2,a2
import numpy as np
weights=[0.15,0.45]
biases=[0.4,0.65]

# Given x1
x1=0.1

#Computing z1 and a1
z1=x1*weights[0]+biases[0]
a1=1.0/(1.0+np.exp(-z1))

# Computing z2 and a2
z2=a1*weights[1]+biases[1]
a2=1.0/(1.0+np.exp(-z2))

with open ("back_propagation_results.txt",'a') as f:
    print('z1 is {}'.format(z1, decimals=3),file=f)
    print('a1 is {}'.format(a1, decimals=3),file=f)
    print('z2 is {}'.format(z2, decimals=3),file=f)
    print('a2 is {}'.format(a2, decimals=3),file=f)

# Info parameters
T=0.25
n_iter=1000
ep=0.001
Lr=0.4

#Differentiation parameters 
from sympy import *
a_2 = Symbol('a_2')
z_2 = Symbol('z_2')
a_1 = Symbol('a_1')
z_1 = Symbol('z_1')
b_2 = Symbol('b_2')
b_1 = Symbol('b_1')
w_2 = Symbol('w_2')
w_1 = Symbol('w_1')
x_1=Symbol('x_1')

E=0.5*((T-a_2)**2)
E_a2_prime=E.diff(a_2)

a_2_exp=1.0/(1.0+exp(-z_2))
a2_z2_prime=a_2_exp.diff(z_2)

z_2_exp1=a1*w_2+biases[1]
z_2_exp2=a1*weights[1]+b_2
z_2_exp3=a_1*weights[1]+biases[1]

z2_w2_prime=z_2_exp1.diff(w_2)
z2_b2_prime=z_2_exp2.diff(b_2)
z2_a1_prime=z_2_exp3.diff(a_1)

a_1_exp=1.0/(1.0+exp(-z_1))
a1_z1_prime=a_1_exp.diff(z_1)

z_1_exp1=x1*w_1+biases[0]
z_1_exp2=x1*weights[0]+b_1

z1_w1_prime=z_1_exp1.diff(w_1)
z1_b1_prime=z_1_exp2.diff(b_1)

# Calculation the differentation

E=lambdify(a_2, E)
E_a2_prime=lambdify(a_2, E_a2_prime)
a2_z2_prime=lambdify(z_2, a2_z2_prime)
z2_w2_prime=lambdify(w_2, z2_w2_prime)
z2_b2_prime=lambdify(b_2, z2_b2_prime)
z2_a1_prime=lambdify(a_1, z2_a1_prime)
a1_z1_prime=lambdify(z_1, a1_z1_prime)
z1_w1_prime=lambdify(w_1, z1_w1_prime)
z1_b1_prime=lambdify(b_1, z1_b1_prime)
## Updating w2,b2,w1,w2

w2_updated=weights[1]-Lr*(E_a2_prime(a2)*a2_z2_prime(z2)*z2_w2_prime(weights[1]))
b2_updated=biases[1]-Lr*(E_a2_prime(a2)*a2_z2_prime(z2)*z2_b2_prime(biases[1]))
w1_updated=weights[0]-Lr*(E_a2_prime(a2)*a2_z2_prime(z2)*z2_a1_prime(a1)*a1_z1_prime(z1)*z1_w1_prime(weights[0]))
b1_updated=biases[0]-Lr*(E_a2_prime(a2)*a2_z2_prime(z2)*z2_a1_prime(a1)*a1_z1_prime(z1)*z1_b1_prime(biases[0]))

with open ("back_propagation_results.txt",'a') as f:
    print('Updated w2 is {}'.format(w2_updated, decimals=3),file=f)
    print('Updated b2 is {}'.format(b2_updated, decimals=3),file=f)
    print('Updated w1 is {}'.format(w1_updated, decimals=3),file=f)
    print('Updated b1 is {}'.format(b1_updated, decimals=3),file=f)

# Computing new error

#Computing z1_new and a1_new
z1_updated=x1*w1_updated+b1_updated
a1_updated=1.0/(1.0+np.exp(-z1_updated))

# Computing z2 and a2
z2_updated=a1_updated*w2_updated+b2_updated
a2_updated=1.0/(1.0+np.exp(-z2_updated))

E_old=0.5*((T-a2)**2)
E_new=0.5*((T-a2_updated)**2)

with open ("back_propagation_results.txt",'a') as f:
    print('a2 output before single iteration is {}'.format(a2, decimals=3),file=f)
    print('a2 output after single iteration is {}'.format(a2_updated, decimals=3),file=f)
    print('Error before single iteration is {}'.format(E_old, decimals=3),file=f)
    print('Error after single iteration is {}'.format(E_new, decimals=3),file=f)

## Using loops of iteration, the error can be approached to zero to find optimum weights and biases
