# digits
Hand-written digit classification using a Neural Network built from scratch.

# Why digits?
The idea of implementing a neural network from scratch came to me while I was working through Prof. Andrew Ng's Machine Learning course on Coursera.
A lot of concepts involved in neural networks made intuitive sense to me, but I felt I could do more on the mathematical background of it.
Specifically, I wanted to get an idea of how computations can be vectorized with different activation and cost functions.

digits, in particular, uses the Leaky Rectified Linear Unit (LReLU) and the Softmax functions as activation functions for the hidden layers and output layer (respectively),
and the Cross-Entropy Loss / Log Loss function as the cost function. The optimization function is Stochastic Gradient Descent (SGD), and the partial derivatives are computed using Backpropagation. Furthermore, digits also uses preprocessing techniques such as deskewing on the MNIST dataset. Images drawn by the user undergo preprocessing as well, with additional steps such as normalization. 

Aside from implementation, I also manually derived the vectorized formulae for the partial derivatives primarily using dimensional analysis.

### Major Technologies Used
- Python 3
- Numpy
- Matplotlib
- Scipy
- Pillow
- Weights & Biases

<!--
### Things I've Learnt
- Deriving vectorized partial derivatives using dimensional analysis
- Implementing SGD 
-->
