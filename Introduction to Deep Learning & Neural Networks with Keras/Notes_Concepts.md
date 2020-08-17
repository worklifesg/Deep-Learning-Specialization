# Notes on Deep Learning Concepts

## Table of Content
  * [Gradient Descent](#gradient-descent)

### Gradient Descent
  * It is an iterative apporach to find the minimum of the function. Derivative of the function is evaluated and the cost function is minimized to find the optimized parameter. The cost function is defined as:
      
      ![equation](http://www.sciweavers.org/tex2img.php?eq=J%28%20%5Ctheta%20%29%3D%20%5Cdfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20y%5Ei%20log%28%20%5Cwidehat%7By%7D%20%5Ei%29%20%2B%20%281-y%5Ei%29log%281%5Cwidehat%7By%7D%5Ei%29&bc=White&fc=Black&im=jpg&fs=12&ff=modern&edit=0)
  * We try to minimize position on the ERROR curve by adding or deducting weights. The objective is diminish the slope and we descend down by steps till slope approaches zero.
  * Two important parameters in Gradient Descent:
    * Direction - direction of the greatest UPHILL
    * Size of steps - to find the gradient of the curve (gradient means the slope of the surface at every point)
      * Slope of the function: 
        ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cdfrac%7B%5Cpartial%20J%28%20%5Ctheta%20%29%7D%7B%5Cpartial%20%5Ctheta%7D&bc=White&fc=Black&im=jpg&fs=12&ff=modern&edit=0)

      * Large Slope - large steps - ! might miss the optimum value
      * Small Slope - small steps - ! might take long time to find the optimum value.
  * Discussing another way to find error (MSE) for a simple line equation is given as:
  
    ![equation](http://www.sciweavers.org/tex2img.php?eq=J%28%5Ctheta%29%3D%5Cdfrac%7B1%7D%7B2m%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%5Cbigg%28z_i-%5Ctheta%20x_i%20%5Cbigg%29%5E2&bc=White&fc=Black&im=jpg&fs=12&ff=modern&edit=0)
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
   ![equation](http://www.sciweavers.org/tex2img.php?eq=%20%5Ceta%20%5Cdfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%7D&bc=White&fc=Black&im=jpg&fs=12&ff=modern&edit=0)
  * Note: At each iteration, the weight of the function is updated with proportion to the negative of the gradient of the function at the current point.
