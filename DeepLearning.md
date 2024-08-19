- AI: 
The capability of a machine to imitate the intelligent  human behavior
- ML: 
explores algorithms that learn from data. Builds models from data 
without being explicitly programmed.
- DL: 
Subset of ML which mostly works on big datasets. 

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
Sigmoid Activation Function The sigmoid function is defined as: `σ(x)=1/1+e^σ(x)^T​` This function has an S-shaped curve, which helps in introducing non-linearity into the model, allowing the neural network to learn more complex patterns


# 4. Gradient Descent 
is an iterative first order optimization algorithm that finds localminima by differentiating a given cost function. Gradient descent aims to find the minimum of a cost function, which measures the difference between the predicted and actual values. The algorithm updates the model’s parameters (weights and biases) to reduce this error.


# 2. Feedforward Neural Network: 
is a type of artificial neural network where connections between the nodes do not form cycles. This characteristic differentiates it from recurrent neural networks (RNNs). The network consists of an input layer, one or more hidden layers, and an output layer. Information flows in one direction—from input to output—hence the name “feedforward.”
Structure of a Feedforward Neural Network
- Input Layer: 
consists of neurons that receive the input data. Each neuron in the input layer represents a feature of the input data.
- Hidden Layers: 
One or more hidden layers are placed between the input and output layers. These layers are responsible for learning the complex patterns in the data. Each neuron in a hidden layer applies a weighted sum of inputs followed by a non-linear activation function.
- Output Layer: 
provides the final output of the network. The number of neurons in this layer corresponds to the number of classes in a classification problem or the number of outputs in a regression problem.


