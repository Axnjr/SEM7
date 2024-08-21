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
- Universal Approximators:
It has been theoretically proven that MLPs with a single hidden layer containing a sufficient number of neurons (under certain conditions) can approximate  any continuous function to arbitrary accuracy. 
- Hierarchy of Features: 
MLPs can learn hierarchical representations of data. Each layer in 
an MLP learns progressively more abstract features from the previous layer, allowing the network to discover and utilize multiple levels of abstraction in the data.
- Depth and Expressiveness: 
Deeper networks can potentially capture more nuanced(subtle and detailed differences ) and complex relationships in data, though they may require more computational resources and careful tuning


# 3. Sigmoid neuron: 
is similar to a perceptron but uses a sigmoid activation function. This function maps any input value to a value between 0 and 1, which is useful for binary classification tasks. <br>
Sigmoid Activation Function is defined as: `σ(x)=1/1+e^σ(x)^T​` This function has an S-shaped curve, which helps in introducing non-linearity into the model, allowing the neural network to learn more complex patterns

# 4. Loss functions:
A loss function is a function that compares the target and the predicted output values. 
While training, we aim to minimize the loss between the predicted and target outputs. <br>
The loss function is also known as error function. <br>
A loss function applies to a single training example, whereas a cost function (or sometimes called as objective function) is an average of the loss function of an entire training set containing several training examples. Type:
 1. Classification loss function (for discrete numeric values): e.g. Binary cross entropy loss, categorical Mean Absolute Error (MAE) / L1 Loss cross entropy loss, hinge loss, log loss.
 2. Regression loss function (for continuous numeric values): e.g. Mean squared error/L2 loss, Mean absolute error/L1 loss, Huber loss/smooth mean absolute error.

# 5. Gradient Descent 
is an iterative first order optimization algorithm that finds local minima by differentiating a given cost function. Gradient descent aims to find the minimum of a cost function, which measures the difference between the predicted and actual values. The algorithm updates the model’s parameters (weights and biases) to reduce this error.

Gradient descent has an analogy in which we have to imagine ourselves at the top of a mountain valley and left stranded and blindfolded, our objective is to reach the bottom of the hill. Feeling the slope of the terrain around you is what everyone would do. Well, this action is analogous to calculating the gradient descent, and taking a step is analogous to one iteration of the update to the parameters.

Choosing a perfect learning rate is a very important task as it depends on how large of a step we take downhill during each iteration.
If we take too large of a step, we may step over the minimum. However, if we take small steps, it will require many iterations to arrive at the minimum.

# 6. Challenges and Considerations
- Learning Rate:     
Choosing an appropriate learning rate is crucial. A high learning rate can cause the algorithm to overshoot the minimum, while a low learning rate can make the convergence process slow.

- Local Minima and Saddle Points: 
Gradient descent can get stuck in local minima or saddle points, especially in non-convex cost functions common in deep learning.

- Vanishing and Exploding Gradients: 
In deep networks, gradients can become very small (vanishing) or very large (exploding), making training difficult. Techniques like gradient clipping and normalization can help mitigate these issues.

# 7. Types of Gradient Descent
- Batch Gradient Descent: 
Uses the entire training dataset to compute the gradient and update the parameters. It is computationally expensive for large datasets but provides a stable convergence.

- Mini-Batch Gradient Descent: 
A compromise between batch and stochastic gradient descent. It uses a small batch of training examples to compute the gradient and update the parameters. It balances the efficiency and stability of the updates.
<img width="218" alt="image" src="https://github.com/user-attachments/assets/1f5e8a4b-10be-4422-a000-95870c35e4c8">

- Stochastic Gradient Descent (SGD): 
Uses a single training example to compute the gradient and update the parameters. It is faster and can escape local minima but introduces more noise in the updates.
<img width="272" alt="image" src="https://github.com/user-attachments/assets/49d6ac6a-f1d8-4c2f-a4d9-aeb73dbed6a6">
	<img width="265" alt="image" src="https://github.com/user-attachments/assets/20a34c74-cbb2-41f9-8710-ca3a5e59bde9">

- Momentum gradient descent: 
enhances the standard gradient descent by adding a momentum term. This helps accelerate the convergence of the training process, reduces oscillations and better handle the local minima / smooth out the updates.
	<img width="350" alt="image" src="https://github.com/user-attachments/assets/f7e9e64a-de87-4de7-b5c8-d7f173dedab7">

- Nesterov Accelerated GD (NAG):
NAG modifies the Momentum-based Gradient Descent by calculating the gradient not at the current parameters but with a look-ahead based on the velocity.
	1. Look-Ahead: Instead of calculating the gradient at the current parameters, NAG first performs a look-ahead step to estimate where the 		parameters will be if the current velocity were applied.
	2. Gradient Calculation: The gradient is then computed at this look-ahead point, providing a more accurate estimate of the direction in which the 	parameters should be updated.
	3. Velocity Update: The velocity term is updated using this more accurate gradient, making the updates more informed and potentially more 		efficient.
	4. Parameter Update: Finally, the parameters are updated using the updated velocity.
	- <img width="317" alt="image" src="https://github.com/user-attachments/assets/746bb2e1-7849-458e-b442-c247eeeedb50">
	- By considering the future position of the parameters, NAG often converges faster than momentum-based gradient descent.
	- The look-ahead mechanism provides more informed updates, which can lead to better convergence properties

- AdaGrad:
is an optimization algorithm designed to adapt the learning rate for each parameter individually based on the historical gradients. This adaptive nature allows AdaGrad to perform well in scenarios with sparse data and features, where different parameters may have different  degrees of importance and frequency.
  - Key Concepts
	1. Adaptive Learning Rate: Unlike traditional gradient descent, which uses a single learning rate for all parameters, AdaGrad adjusts the 		learning rate for each parameter dynamically.
	2. Accumulation of Squared Gradients: AdaGrad keeps track of the sum of the squares of the gradients for each parameter. This accumulated value 	is then used to adjust the learning rate.
  - <img width="446" alt="image" src="https://github.com/user-attachments/assets/4c63dfc5-2e70-498b-b97a-3fb775f250f3">
  -  Advantages
		1. Adaptivity: Automatically adjusts learning rates for each parameter, making it effective for problems with sparse features.
		2. Stability: Reduces the learning rate over time for frequently updated parameters, which can help stabilize convergence.
  - Disadvantages: Aggressive Decay: For some problems, the learning rate might decay too aggressively, causing the learning process to stop too early 	 before reaching the optimal solution

- RMSProp (Root Mean Square Propagation):
is an adaptive learning rate optimization algorithm designed to address some of the limitations of AdaGrad, particularly the issue of rapidly decaying learning rates. RMSProp aims to maintain a balance by controlling the learning rate decay, which allows for more stable and faster convergence, especially in deep learning applications.

- Adam(Adaptive Moment Estimation):
is an optimization algorithm that combines the best properties of the AdaGrad and RMSProp algorithms to provide an efficient and adaptive learning rate. It is particularly well-suited for problems involving large datasets and highdimensional parameter spaces

# 7. Activation functions: 
in neural networks is a mathematical function applied to the output of a neuron. Its primary purpose is to introduce non-linearity into the model, enabling the network to learn and represent complex patterns in the data. Here are four common activation functions:
- The sigmoid function: 
maps any input to a value between 0 and 1. It’s often used in binary classification problems. `y = 1 / 1 + e ^ -x`
	- Pros: Smooth gradient, output range (0, 1).
	- Cons: Can cause vanishing gradient problems, slow convergence.
- Tanh function: 
is very similar to the sigmoid/logistic activation function, and even has the same S-shape with the difference in output range of -1 to 		1. In Tanh, the larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more 			negative), the closer 	the output will be to - 1.0. `f(x) = (ex - e-x) / (ex +e-x)`
-  ReLU:
is one of the most popular activation functions. It outputs the input directly if it is positive; otherwise, it outputs zero. `f(x) = max(0, x)`
	- Pros: Computationally efficient, helps mitigate the vanishing gradient problem.
	- Cons: Can cause “dying ReLU” problem where neurons can become inactive.
- Leaky ReLU (Leaky Rectified Linear Unit):
is a variation of the ReLU activation function designed to address the “dying ReLU” problem, where neurons can become inactive and only output zero for any input. `f(x) = max(0.1x, x)`


# 8. Dropout: 
is a regularization technique used in deep learning to prevent overfitting. During training, dropout randomly deactivates a fraction of neurons in a layer, effectively creating a sparse network. This randomness forces the network to learn more robust features that are not overly reliant on specific neurons.
- Random Deactivation: 
During each training iteration, a specified percentage of neurons are randomly “dropped out” or deactivated.
- Scaling: 
The outputs of the remaining active neurons are scaled up to maintain the overall output level.
- Reactivation: 
In the next iteration, a different set of neurons may be deactivated.
#### Advantages:
- Prevents Overfitting: 
By not allowing the network to rely too heavily on any single neuron, dropout reduces the risk of overfitting to the training data2.
- Ensemble Effect: 
Dropout simulates training multiple different neural networks and averaging their predictions, which improves generalization1.
- Improved Data Representation: 
The introduction of noise through dropout helps the network learn more generalized features


# 7. Feedforward Neural Network: 
is a type of artificial neural network where connections between the nodes do not form cycles. This characteristic differentiates it from recurrent neural networks (RNNs). The network consists of an input layer, one or more hidden layers, and an output layer. Information flows in one direction—from input to output—hence the name “feedforward.”
Structure of a Feedforward Neural Network
- Input Layer: 
consists of neurons that receive the input data. Each neuron in the input layer represents a feature of the input data.
- Hidden Layers: 
One or more hidden layers are placed between the input and output layers. These layers are responsible for learning the complex patterns in the data. Each neuron in a hidden layer applies a weighted sum of inputs followed by a non-linear activation function.
- Output Layer: 
provides the final output of the network. The number of neurons in this layer corresponds to the number of classes in a classification problem or the number of outputs in a regression problem.


