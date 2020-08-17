# Notes on Deep Learning Concepts

## Table of Content
  * [Gradient Descent](#gradient-descent)

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
