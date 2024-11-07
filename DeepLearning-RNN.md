# 1. The sequence learning problem in deep learning: 
involves predicting the next element in a sequence based on previous elements. This is particularly useful in applications like natural language processing (NLP), speech recognition, time series prediction, and music generation.

# 2. Unfolding computational graphs: 
is a technique used to represent and visualize the structure of computations, especially in the context of recurrent neural networks (RNNs) and time series data. By unfolding a computational graph, we can transform a recursive or recurrent computation into a sequence of operations that are easier to analyze and optimize. (`Computational graphs are a type of graph that can be used to represent mathematical expressions.` This is similar to descriptive language in the case of deep learning models, providing a functional description of the required computation.)

# 3. Drawbacks of standard neural network
- Standard neural architecture will not perform well for sequence
models
- Feed fwd network accepts a fixed size vector as the input and
produce a fixed size vector as the output
- Does not share the features learned across the different
positions of the text
- Sequence and length has to be maintained in a network for
further processing

# 4. Feed forward vs RNN:
![alt text](image-6.png)

# 5. Recurrent Neural Networks (RNNs): 
are a type of neural network designed specifically for `sequential data`. They are widely used in tasks where the order of the data points matters, such as time series analysis, natural language processing, and speech recognition. It is a type of Neural Network where the output from the previous step is fed as input to the current step. The main and `most important feature of RNN is its Hidden state`, which remembers some information about a sequence. The state is also referred to as Memory State since it remembers the previous input to the network. The fundamental processing unit in a `Recurrent Neural Network (RNN)` is a `Recurrent Unit`.

### Types Of RNN
There are four types of RNNs based on the number of inputs and outputs in the network.
- One to One 
- One to Many 
- Many to One 
- Many to Many 

### Key Components of an RNN Architecture

- **Input Layer:** 
accepts sequential data. Each step in the sequence has a set of features (for example, words in a sentence or data points in a time series).
- **Hidden Layer (Recurrent Layer):**
The hidden layer is the core of the RNN, where the recurrent connections are present.
Each hidden state `ℎ𝑡`​at time step `𝑡` depends not only on the current input `𝑥𝑡` ​but also on the previous hidden state `ℎ𝑡 − 1`, creating a `feedback loop`.
The recurrent layer uses an activation function, often a `hyperbolic tangent (tanh) or ReLU`.

- **Output Layer:**
The output layer generates predictions, which can vary based on the problem.
In classification, it might have a softmax activation to produce probabilities for different classes. For regression, it might have a linear activation function.
- **Weight Sharing:**
RNNs share the same weights across all time steps, making them efficient for long sequences.

### Key Equations in a Basic RNN For a given time step `𝑡`, Hidden State Update:
### `ℎ𝑡 = 𝑓(𝑊𝑥ℎ * 𝑥𝑡 + 𝑊ℎℎ * ℎ𝑡−1 + 𝑏ℎ)`, **Where:** (**xh, hh, hy, t, h are in subscripts**)
- `ht` is the current hidden state.
- `𝑊𝑥ℎ` and `𝑊ℎℎ` are the weight matrices for the input-to-hidden and hidden-to-hidden connections 
- `bh` is the bias term
- `f` is the activation function (often tanh or ReLU).

### Output Generation: `𝑦𝑡 = 𝑔(𝑊ℎ𝑦 * ℎ𝑡 + 𝑏𝑦)`, **where:**<br>
- `yt`is the output at time step `t`,
- `Why` is the weight matrix from hidden to output,
- `by` is the output bias,
- `g` is the output activation function (e.g., softmax for classification).

### Forward Propagation in a Nutshell
During forward propagation, the RNN moves through the sequence one step at a time, updating the hidden state and producing an output at each time step. Each hidden state `ℎ𝑡` contains information from all previous inputs in the sequence, enabling the RNN to make context-aware predictions based on the entire input sequence up to that point. This sequential processing is what allows RNNs to handle tasks with temporal dependencies, such as language and time-series prediction.

### Advantages

- An RNN remembers each and every piece of information through time. 
- It is useful in time series prediction only because of the feature to remember previous inputs as well. This is called Long Short Term Memory.
- Recurrent neural networks are even used with convolutional layers to extend the effective pixel neighborhood.

### Disadvantages

- Gradient vanishing and exploding problems.
- Training an RNN is a very difficult task.
- It cannot process very long sequences if using tanh or relu as an activation function.

### Applications of Recurrent Neural Network
- Language Modelling and Generating Text
- Speech Recognition
- Machine Translation
- Image Recognition, Face detection
- Time series Forecasting

# 6. Limitations of vanilla RNN:
Vanilla Recurrent Neural Networks (RNNs) do face significant limitations, primarily due to the phenomena of vanishing and exploding gradients. These issues are particularly problematic when dealing with long-term dependencies in sequential data. Here’s a closer look:

### Vanishing Gradients
- **Definition:** Occurs when the gradients used to update the weights during training become exceedingly small, effectively preventing the network from learning.
- **Consequence:** The network struggles to learn long-range dependencies because the earlier layers receive almost negligible gradient updates.
- **Why It Happens:** In each layer, gradients are multiplied by the weights. If these weights are small, the gradients exponentially decrease as they propagate back through the layers.

### Exploding Gradients
- **Definition:** When the gradients grow excessively large, causing the weights to update too drastically, which can lead to network instability.
- **Consequence:** This can result in numerical overflow and erratic changes in the weights, making the model difficult to train and causing it to diverge.
- **Why It Happens:** If the weights are large, the gradients can grow exponentially as they are propagated back through the network, leading to overflow.

### Impact on Vanilla RNNs

- **Training Difficulty:** Both vanishing and exploding gradients make training vanilla RNNs challenging, especially for tasks that require learning from long sequences of data.
- **Performance:** These issues significantly impact the model's ability to retain information from earlier time steps, resulting in poor performance on tasks that involve long-term dependencies.

### Mitigation Strategies

- **Gradient Clipping:** Limiting the size of gradients to prevent them from getting too large.
- **LSTM and GRU:** Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) are advanced variants of RNNs designed to mitigate these issues by maintaining more stable gradients over long sequences.

# 7. Bidirectional Recurrent Neural Networks (BRNNs) 
is an extension of the standard RNN that processes sequential data in both forward and backward directions. This architecture allows the network to access both past and future information about a particular time step, making it more effective for tasks where context from both directions is useful.

### Key Characteristics of BRNNs

- BRNNs have two RNN layers: 
one processes the input sequence in a forward direction (from start to end), and the other processes it in a backward direction (from end to start).
Each time step’s output is influenced by information from both the previous and future states.

- Concatenated Outputs:
At each time step, the hidden states from both the forward and backward RNNs are concatenated (or otherwise combined), allowing the BRNN to generate a more contextually rich representation of each time step. This combined representation is then used as input to the output layer for predictions.

- For each time step `𝑡`, combine the forward and backward hidden states, resulting in a bidirectional representation 
`ℎ𝑡 = [ℎ𝑡forward ; ℎ𝑡backward]`
Output Generation: Use this combined hidden state to produce the output `𝑦𝑡` at each time step.