# Notes on Deep Learning Concepts

## Table of Content
  * [Gradient Descent](#gradient-descent)
  * [Backpropagation](#backpropagation)

### Gradient Descent
  * It is an iterative apporach to find the minimum of the function. Derivative of the function is evaluated and the cost function is minimized to find the optimized parameter. The cost function is defined as:
      
      ![J(\theta)=-\dfrac{1}{m} \sum_{i=1}^m y^i \log (\hat{y}^i) + (1-y^i) \log (1-\hat{y}^i)](https://latex.codecogs.com/svg.latex?J(\theta)=-\dfrac{1}{m}&space;\sum_{i=1}^m&space;y^i&space;\log&space;(\hat{y}^i)&space;&plus;&space;(1-y^i)&space;\log&space;(1-\hat{y}^i))
  * We try to minimize position on the ERROR curve by adding or deducting weights. The objective is diminish the slope and we descend down by steps till slope approaches zero.
  * Two important parameters in Gradient Descent:
    * Direction - direction of the greatest UPHILL
    * Size of steps - to find the gradient of the curve (gradient means the slope of the surface at every point)
      * Slope of the function: 
        ![\dfrac{\partial J(\theta)}{\partial \theta}](https://latex.codecogs.com/svg.latex?\dfrac{\partial&space;J(\theta)}{\partial&space;\theta})

      * Large Slope - large steps - ! might miss the optimum value
      * Small Slope - small steps - ! might take long time to find the optimum value.
  * Discussing another way to find error (MSE) for a simple line equation is given as:
  
    ![J(\theta)=\dfrac{1}{2m} \sum_{i=1}^m \bigg(z_i-\theta x_i \bigg)^2 ](https://latex.codecogs.com/svg.latex?J(\theta)=\dfrac{1}{2m}&space;\sum_{i=1}^m&space;\bigg(z_i-\theta&space;x_i&space;\bigg)^2)
    * The best value of theta will be value that results in minimum value of the cost function. This value can be optimized using Gradient Descent Algorithm as follows:
    
      ```
       1. Intialize theta with random value as theta_0, let us say as 0.2
       2. Start taking steps towards the minimum value of the function (mostly approaches to zero or the minimum value of the cost function - can be found by derivative of the function.
        2.1 One takes steps proportional to the negative of the gradient of the function at current point.
       3. The magntiude of the step is controlled by the factor 'Learning Rate'.
       4. Take a step and move to next step, let's say theta_1.
       5. We repeat these steps till we achieve the function minimum
      ```
     
   Learning Rate:
   ![\eta \dfrac{\partial&space;J(\theta)}{\partial&space;\theta}](https://latex.codecogs.com/svg.latex?\eta&space;\dfrac{\partial&space;J(\theta)}{\partial&space;\theta})
   
   ![\theta_1=\theta_0-\eta\dfrac{\partial J(\theta)}{\partial \theta}](https://latex.codecogs.com/svg.latex?\theta_1=\theta_0-\eta\dfrac{\partial&space;J(\theta)}{\partial&space;\theta})
  * Note: At each iteration, the weight of the function is updated with proportion to the negative of the gradient of the function at the current point.

### Backpropagation
 * While working on forward propagation (process through which data passes through layers of neurons in neural network from input to output layer), it is assumed that the weights and biases are OPTIMIZED but in practice, they are NOT OPTIMIZED.
 * Thus we use both Forward and Back propagation to Train and Optimize weights and biases in dataset.
 * In practive the output (a<sub>n</sub>) and ground truth (T) are different and we use Backpropagation Algorithm to minimize the error between these two parameters:
 
    ```
       1. Calculate Error between T and a<sub>n</sub> (not cost function or error function)
       2. Propagate this error back into the network and perform Gradient Descent of the weights and biases to optimize them.
    ```
 * Example - One input with two hidden layers and one output as shown in figure below


<p align="center">
  <img width="600" alt="java 8 and prio java 8  array review example" img align="center" src ="https://github.com/worklifesg/Deep-Learning-Specialization-In-Progress-/blob/master/images/Back_propagation.png">
</p> 

 
 
     Error For one data point:     
    
   ![E=\dfrac{1}{2}\bigg(T-a_2\bigg)^2](https://latex.codecogs.com/gif.latex?E=\dfrac{1}{2}\bigg(T-a_2\bigg)^2)
   
     Error For multiple data points:     
    
   ![E=\dfrac{1}{2m}\sum_{i=1}^m\bigg(T_i-a_{2,i}\bigg)^2](https://latex.codecogs.com/gif.latex?E=\dfrac{1}{2m}\sum_{i=1}^m\bigg(T_i-a_{2,i}\bigg)^2)
   
    Update weights and biases
 
   ![w_i\longrightarrow w_i-\eta . \dfrac{\partial E}{\partial w_i}](https://latex.codecogs.com/gif.latex?w_i\longrightarrow&space;w_i-\eta&space;.&space;\dfrac{\partial&space;E}{\partial&space;w_i})             
   
   ![b_i\longrightarrow b_i-\eta . \dfrac{\partial E}{\partial b_i}](https://latex.codecogs.com/gif.latex?b_i\longrightarrow&space;b_i-\eta&space;.&space;\dfrac{\partial&space;E}{\partial&space;b_i})

    Example - calculation of w2 update
   
   ![\begin{align*}
E&=\dfrac{1}{2}\bigg(T-a_2\bigg)^2 \longrightarrow \dfrac{\partial E}{\partial a_2}\\
a_2&=f(z_2)=\dfrac{1}{1+e^{-z_2}} \longrightarrow \dfrac{\partial a_2}{\partial z_2}\\
z_2&=a_1w_2+b_2 \longrightarrow \dfrac{\partial z_2}{\partial w_2}\\
\qquad\\
w_2&= w_2-\eta . \dfrac{\partial E}{\partial w_2}\\
&=w_2-\eta . \dfrac{\partial E}{\partial a_2}\dfrac{\partial a_2}{\partial z_2}\dfrac{\partial z_2}{\partial w_2}\\
&=w_2-\eta . \bigg[-(T-a_2).(a_2(1-a_2)).(a_1)\bigg]
\end{align*}](https://latex.codecogs.com/gif.latex?\begin{align*}&space;E&=\dfrac{1}{2}\bigg(T-a_2\bigg)^2&space;\longrightarrow&space;\dfrac{\partial&space;E}{\partial&space;a_2}\\&space;a_2&=f(z_2)=\dfrac{1}{1&plus;e^{-z_2}}&space;\longrightarrow&space;\dfrac{\partial&space;a_2}{\partial&space;z_2}\\&space;z_2&=a_1w_2&plus;b_2&space;\longrightarrow&space;\dfrac{\partial&space;z_2}{\partial&space;w_2}\\&space;\qquad\\&space;w_2&=&space;w_2-\eta&space;.&space;\dfrac{\partial&space;E}{\partial&space;w_2}\\&space;&=w_2-\eta&space;.&space;\dfrac{\partial&space;E}{\partial&space;a_2}\dfrac{\partial&space;a_2}{\partial&space;z_2}\dfrac{\partial&space;z_2}{\partial&space;w_2}\\&space;&=w_2-\eta&space;.&space;\bigg[-(T-a_2).(a_2(1-a_2)).(a_1)\bigg]&space;\end{align*})

    Example - calculation of b2 update
   
   ![\begin{align*}
E&=\dfrac{1}{2}\bigg(T-a_2\bigg)^2 \longrightarrow \dfrac{\partial E}{\partial a_2}\\
a_2&=f(z_2)=\dfrac{1}{1+e^{-z_2}} \longrightarrow \dfrac{\partial a_2}{\partial z_2}\\
z_2&=a_1w_2+b_2 \longrightarrow \dfrac{\partial z_2}{\partial b_2}\\
\qquad\\
b_2&= b_2-\eta . \dfrac{\partial E}{\partial b_2}\\
&=b_2-\eta . \dfrac{\partial E}{\partial a_2}\dfrac{\partial a_2}{\partial z_2}\dfrac{\partial z_2}{\partial b_2}\\
&=b_2-\eta . \bigg[-(T-a_2).(a_2(1-a_2)).(1)\bigg]
\end{align*}](https://latex.codecogs.com/gif.latex?\begin{align*}&space;E&=\dfrac{1}{2}\bigg(T-a_2\bigg)^2&space;\longrightarrow&space;\dfrac{\partial&space;E}{\partial&space;a_2}\\&space;a_2&=f(z_2)=\dfrac{1}{1&plus;e^{-z_2}}&space;\longrightarrow&space;\dfrac{\partial&space;a_2}{\partial&space;z_2}\\&space;z_2&=a_1w_2&plus;b_2&space;\longrightarrow&space;\dfrac{\partial&space;z_2}{\partial&space;b_2}\\&space;\qquad\\&space;b_2&=&space;b_2-\eta&space;.&space;\dfrac{\partial&space;E}{\partial&space;b_2}\\&space;&=b_2-\eta&space;.&space;\dfrac{\partial&space;E}{\partial&space;a_2}\dfrac{\partial&space;a_2}{\partial&space;z_2}\dfrac{\partial&space;z_2}{\partial&space;b_2}\\&space;&=b_2-\eta&space;.&space;\bigg[-(T-a_2).(a_2(1-a_2)).(1)\bigg]&space;\end{align*})


     Example - calculation of w1 update
   
   ![\begin{align*}
E&=\dfrac{1}{2}\bigg(T-a_2\bigg)^2 \longrightarrow \dfrac{\partial E}{\partial a_2}\\
a_2&=f(z_2)=\dfrac{1}{1+e^{-z_2}} \longrightarrow \dfrac{\partial a_2}{\partial z_2}\\
z_2&=a_1w_2+b_2 \longrightarrow \dfrac{\partial z_2}{\partial a_1}\\
\qquad\\
a_1&=f(z_1)=\dfrac{1}{1+e^{-z_1}} \longrightarrow \dfrac{\partial a_1}{\partial z_1}\\
z_1&=x_1w_1+b_1 \longrightarrow \dfrac{\partial z_1}{\partial w_1}\\
\qquad\\
w_1&= w_1-\eta . \dfrac{\partial E}{\partial w_1}\\
&=w_1-\eta . \dfrac{\partial E}{\partial a_2}\dfrac{\partial a_2}{\partial z_2}\dfrac{\partial z_2}{\partial a_1}\dfrac{\partial a_1}{\partial z_1}\dfrac{\partial z_1}{\partial w_1}\\
&=w_1-\eta . \bigg[-(T-a_2).(a_2(1-a_2)).(w_2).(a_1(1-a_1)).x_1\bigg]
\end{align*}](https://latex.codecogs.com/gif.latex?\begin{align*}&space;E&=\dfrac{1}{2}\bigg(T-a_2\bigg)^2&space;\longrightarrow&space;\dfrac{\partial&space;E}{\partial&space;a_2}\\&space;a_2&=f(z_2)=\dfrac{1}{1&plus;e^{-z_2}}&space;\longrightarrow&space;\dfrac{\partial&space;a_2}{\partial&space;z_2}\\&space;z_2&=a_1w_2&plus;b_2&space;\longrightarrow&space;\dfrac{\partial&space;z_2}{\partial&space;a_1}\\&space;\qquad\\&space;a_1&=f(z_1)=\dfrac{1}{1&plus;e^{-z_1}}&space;\longrightarrow&space;\dfrac{\partial&space;a_1}{\partial&space;z_1}\\&space;z_1&=x_1w_1&plus;b_1&space;\longrightarrow&space;\dfrac{\partial&space;z_1}{\partial&space;w_1}\\&space;\qquad\\&space;w_1&=&space;w_1-\eta&space;.&space;\dfrac{\partial&space;E}{\partial&space;w_1}\\&space;&=w_1-\eta&space;.&space;\dfrac{\partial&space;E}{\partial&space;a_2}\dfrac{\partial&space;a_2}{\partial&space;z_2}\dfrac{\partial&space;z_2}{\partial&space;a_1}\dfrac{\partial&space;a_1}{\partial&space;z_1}\dfrac{\partial&space;z_1}{\partial&space;w_1}\\&space;&=w_1-\eta&space;.&space;\bigg[-(T-a_2).(a_2(1-a_2)).(w_2).(a_1(1-a_1)).x_1\bigg]&space;\end{align*})

    Example - calculation of b1 update
   
   ![\begin{align*}
E&=\dfrac{1}{2}\bigg(T-a_2\bigg)^2 \longrightarrow \dfrac{\partial E}{\partial a_2}\\
a_2&=f(z_2)=\dfrac{1}{1+e^{-z_2}} \longrightarrow \dfrac{\partial a_2}{\partial z_2}\\
z_2&=a_1w_2+b_2 \longrightarrow \dfrac{\partial z_2}{\partial a_1}\\
\qquad\\
a_1&=f(z_1)=\dfrac{1}{1+e^{-z_1}} \longrightarrow \dfrac{\partial a_1}{\partial z_1}\\
z_1&=x_1w_1+b_1 \longrightarrow \dfrac{\partial z_1}{\partial b_1}\\
\qquad\\
b_1&= b_1-\eta . \dfrac{\partial E}{\partial b_1}\\
&=b_1-\eta . \dfrac{\partial E}{\partial a_2}\dfrac{\partial a_2}{\partial z_2}\dfrac{\partial z_2}{\partial a_1}\dfrac{\partial a_1}{\partial z_1}\dfrac{\partial z_1}{\partial b_1}\\
&=b_1-\eta . \bigg[-(T-a_2).(a_2(1-a_2)).(w_2).(a_1(1-a_1)).1\bigg]
\end{align*}](https://latex.codecogs.com/gif.latex?\begin{align*}&space;E&=\dfrac{1}{2}\bigg(T-a_2\bigg)^2&space;\longrightarrow&space;\dfrac{\partial&space;E}{\partial&space;a_2}\\&space;a_2&=f(z_2)=\dfrac{1}{1&plus;e^{-z_2}}&space;\longrightarrow&space;\dfrac{\partial&space;a_2}{\partial&space;z_2}\\&space;z_2&=a_1w_2&plus;b_2&space;\longrightarrow&space;\dfrac{\partial&space;z_2}{\partial&space;a_1}\\&space;\qquad\\&space;a_1&=f(z_1)=\dfrac{1}{1&plus;e^{-z_1}}&space;\longrightarrow&space;\dfrac{\partial&space;a_1}{\partial&space;z_1}\\&space;z_1&=x_1w_1&plus;b_1&space;\longrightarrow&space;\dfrac{\partial&space;z_1}{\partial&space;b_1}\\&space;\qquad\\&space;b_1&=&space;b_1-\eta&space;.&space;\dfrac{\partial&space;E}{\partial&space;b_1}\\&space;&=b_1-\eta&space;.&space;\dfrac{\partial&space;E}{\partial&space;a_2}\dfrac{\partial&space;a_2}{\partial&space;z_2}\dfrac{\partial&space;z_2}{\partial&space;a_1}\dfrac{\partial&space;a_1}{\partial&space;z_1}\dfrac{\partial&space;z_1}{\partial&space;b_1}\\&space;&=b_1-\eta&space;.&space;\bigg[-(T-a_2).(a_2(1-a_2)).(w_2).(a_1(1-a_1)).1\bigg]&space;\end{align*})
