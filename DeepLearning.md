# Recap

- AI: 
The capability of a machine to imitate the intelligent  human behavior
- ML: 
explores algorithms that learn from data. Builds models from data 
without being explicitly programmed.
- DL: 
Subset of ML which mostly works on big datasets. 
- The bias-variance trade-off: 
is a delicate balance between two types of errors:

	- Bias: 
		The difference between the model’s predictions and the true values (high bias leads to underfitting).
		A model with high bias tends to make simplistic assumptions about the data and may underfit the training data.
	- Variance: 
		The variability of model predictions for different training datasets (high variance leads to overfitting).
		A model with high variance is sensitive to small fluctuations in the training data and may overfit the training data.

    - An ideal model strikes a balance between bias and variance.   Increasing model complexity reduces bias but increases variance, and vice versa. The goal is to find the sweet spot where both bias and variance are minimized, leading to optimal generalization.

<br>

# 1. Deep learning Evolution
has evolved significantly over the years, marked by several key milestones and trends:

- Early Developments: 
Originating in the 1960s with artificial neural networks (ANNs), inspired by the human brain’s structure.

- AI Winter: 
In the 1970s and 1980s, progress stalled due to limited computational power, insufficient data, and theoretical challenges, leading to decreased interest and funding.

- Backpropagation: 
Rediscovered in the 1980s, this algorithm allowed efficient training of multi-layer neural networks, overcoming some earlier limitations.

- Convolutional Neural Networks (CNNs): 
Gained prominence in the late 1990s and early 2000s, revolutionizing image recognition tasks with architectures like LeNet-5.

- Big Data and GPUs: 
The early 2010s saw a turning point with the advent of big data and powerful GPUs, enabling large-scale neural network training.

- ImageNet and Deep Learning Renaissance: 
The 2012 ImageNet challenge, won by AlexNet, brought deep learning into the spotlight, sparking widespread research and application.

- Deep Learning in NLP: 
Techniques like RNNs and transformer models (e.g., LSTM, BERT) advanced natural language processing tasks significantly.

- Generative Models: 
Introduction of models like VAEs and GANs opened new possibilities for generating realistic images, videos, and audio.

- Transfer Learning and Pretraining: 
Became prevalent, allowing models to leverage pretraining on large datasets and fine-tune on specific tasks, improving performance and reducing training time.

- Explainability and Interpretability: 
As models grew more complex, researchers focused on making them more understandable and interpretable.


# 2. The representation power of Multi-Layer Perceptrons:
refers to their ability to approximate complex functions and learn non-linear relationships within data.
- Non-linearity: 
MLPs can model non-linear relationships between inputs and outputs. This 
is achieved through the use of non-linear activation functions (e.g., ReLU, sigmoid, tanh) in their hidden layers, allowing them to capture and represent complex patterns in the data.
- Hierarchy of Features: 
MLPs can learn hierarchical representations of data. Each layer in 
an MLP learns progressively more abstract features from the previous layer, allowing the network to discover and utilize multiple levels of abstraction in the data.
- Depth and Expressiveness: 
Deeper networks can potentially capture more nuanced(subtle and detailed differences ) and complex relationships in data, though they may require more computational resources and careful tuning


# 3. Sigmoid neuron: 
is similar to a perceptron but uses a sigmoid activation function. This function maps any input value to a value between 0 and 1, which is useful for binary classification tasks. <br>
Sigmoid Activation Function is defined as: `σ(x)=1/1+e^σ(x)^T​` This function has an S-shaped curve, which helps in introducing non-linearity into the model, allowing the neural network to learn more complex patterns


# 4. Gradient Descent 
is an iterative first order optimization algorithm that finds local minima by differentiating a given cost function. Gradient descent aims to find the minimum of a cost function, which measures the difference between the predicted and actual values. The algorithm updates the model’s parameters (weights and biases) to reduce this error.

Gradient descent has an analogy in which we have to imagine ourselves at the top of a mountain valley and left stranded and blindfolded, our objective is to reach the bottom of the hill. Feeling the slope of the terrain around you is what everyone would do. Well, this action is analogous to calculating the gradient descent, and taking a step is analogous to one iteration of the update to the parameters.

Choosing a perfect learning rate is a very important task as it depends on how large of a step we take downhill during each iteration.
If we take too large of a step, we may step over the minimum. However, if we take small steps, it will require many iterations to arrive at the minimum.

# 5. Challenges and Considerations
- Learning Rate:     
Choosing an appropriate learning rate is crucial. A high learning rate can cause the algorithm to overshoot the minimum, while a low learning rate can make the convergence process slow.

- Local Minima and Saddle Points: 
Gradient descent can get stuck in local minima or saddle points, especially in non-convex cost functions common in deep learning.

- Vanishing and Exploding Gradients: 
In deep networks, gradients can become very small (vanishing) or very large (exploding), making training difficult. Techniques like gradient clipping and normalization can help mitigate these issues.

# 6. Types of Gradient Descent
- Batch Gradient Descent: 
Uses the entire training dataset to compute the gradient and update the parameters. It is computationally expensive for large datasets but provides a stable convergence.

- Stochastic Gradient Descent (SGD): 
Uses a single training example to compute the gradient and update the parameters. It is faster and can escape local minima but introduces more noise in the updates.

- Mini-Batch Gradient Descent: 
A compromise between batch and stochastic gradient descent. It uses a small batch of training examples to compute the gradient and update the parameters. It balances the efficiency and stability of the updates.

- Momentum gradient descent: 
enhances the standard gradient descent by adding a momentum term. This helps accelerate the convergence of the training process, reduces oscillations and better handle the local minima / smooth out the updates. 


# 7. Feedforward Neural Network: 
is a type of artificial neural network where connections between the nodes do not form cycles. This characteristic differentiates it from recurrent neural networks (RNNs). The network consists of an input layer, one or more hidden layers, and an output layer. Information flows in one direction—from input to output—hence the name “feedforward.”
Structure of a Feedforward Neural Network
- Input Layer: 
consists of neurons that receive the input data. Each neuron in the input layer represents a feature of the input data.
- Hidden Layers: 
One or more hidden layers are placed between the input and output layers. These layers are responsible for learning the complex patterns in the data. Each neuron in a hidden layer applies a weighted sum of inputs followed by a non-linear activation function.
- Output Layer: 
provides the final output of the network. The number of neurons in this layer corresponds to the number of classes in a classification problem or the number of outputs in a regression problem.


