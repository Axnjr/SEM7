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

# 8. Backpropagation Through Time (BPTT): 
is the process used to train Recurrent Neural Networks (RNNs) by applying backpropagation over each time step in a sequence. Unlike feedforward networks, which have a straightforward backpropagation process, RNNs require a more complex approach because they maintain a hidden state that evolves across time steps. BPTT adapts the standard backpropagation algorithm to account for this sequence-based dependency.

### Key Steps in BPTT
- **Unrolling the RNN:**
In BPTT, the RNN is `"unrolled" across the sequence length`, creating a separate `copy` of the network for `each time step`.
This unrolling allows the RNN to be visualized as a feedforward network with one layer for each time step in the sequence.
- **Forward Pass Through Time:**
The forward pass is computed for `each time step` in the sequence, `storing the hidden states and outputs at each step`.
This allows the model to `capture dependencies`, as each hidden state depends on both the current input and the previous hidden state.
- **Backward Pass Through Time:**
After the forward pass, BPTT calculates `gradients` by propagating errors backward `through each time step`, starting from the final time step.
`Gradients are computed for each weight with respect to each hidden state and each output`.
The weight updates are `accumulated` and applied after calculating gradients over all time steps in the sequence.
- **Weight Update:**
Once gradients are calculated for each time step, they are summed and applied to update the shared weights of the network, allowing it to learn dependencies across the entire sequence.

### Challenges in BPTT
- Vanishing and Exploding Gradients
- Computational Cost: BPTT is computationally intensive, as it requires storing and processing multiple copies of the network for each time step.

# 9. Truncated BPTT
To address the challenges of long sequences, Truncated Backpropagation Through Time is often used:
- Instead of backpropagating through the entire sequence, `the sequence is divided into shorter segments`, and BPTT is applied within each segment.
- This reduces computation and mitigates the vanishing/exploding gradient problem while still capturing some dependencies.

# 10. Long Short-Term Memory (LSTM): 
is a type of Recurrent Neural Network (RNN) architecture designed to better capture long-term dependencies in sequential data, addressing the common issues of vanishing and exploding gradients that standard RNNs face. 
it incorporate special gating mechanisms that regulate information flow, making them particularly effective for tasks with long-term dependencies like language modeling, time-series prediction, and more.

### Key Components of an LSTM

An LSTM cell has several components, including three gates `(forget, input, and output)` and a cell state. Together, these elements allow it to selectively retain or discard information, improving the network’s ability to learn dependencies over long sequences.
- **Cell State (`𝐶𝑡`):**
The cell state acts as the `"memory"` of the LSTM, allowing information to flow unchanged across `time-steps` unless modified by the gates. It is crucial for retaining long-term information.
- **Hidden State (`ℎ𝑡`):**
The hidden state is the `short-term output` of the LSTM at each `time-step`. This hidden state is also used as input for predictions and is updated at every time step.
- ### **Gates:**
    - **Forget Gate (`𝑓𝑡`):** Decides which information from the cell state should be kept or discarded. It takes the current input 
    `𝑥𝑡` and previous hidden state `ℎ𝑡−1` ​to produce a value between 0 (forget) and 1 (retain).<br>
        `𝑓𝑡 = 𝜎(𝑊𝑓 ⋅ [ℎ𝑡−1,𝑥𝑡] + 𝑏𝑓)`
    - **Input Gate (`𝑖𝑡`):** Controls how much of the new input `𝑥𝑡` should be added to the cell state. It also uses the current input and previous hidden state to decide what to update.<br>
        `it = 𝜎(𝑊𝑓 ⋅ [ℎ𝑡−1,𝑥𝑡] + 𝑏i)`
    - **Output Gate (`ot`):** It Decides which part of the `cell state` will be in the output.<br>
        `ot = 𝜎(𝑊𝑓 ⋅ [ℎ𝑡−1,𝑥𝑡] + 𝑏o)`

### Pitfalls:
- First, they are `more complicated` than traditional RNNs and
require more training data in order to learn effectively.
- Second, they are `not well-suited` for `learning tasks, such
as prediction or classification` tasks where the input data is not a
sequence.
- Third, LSTMs can be `slow to train` on large datasets. This is due
to the fact that they must learn the parameters of the LSTM
cells, which can be `computationally intensive`.
- Finally, LSTMs may not be appropriate for all types of data. For
example, they may not work well with `highly nonlinear data or
data with a lot of noise`.

LSTMs effectively handle the vanishing gradient problem by controlling the flow of information with gates, allowing gradients to remain stable over long sequences. They can capture both short-term and long-term dependencies, making them suitable for a variety of sequence-based tasks.

# 11. The Gated Recurrent Unit (GRU): 
is a type of Recurrent Neural Network (RNN) architecture that was introduced to address some of the limitations of traditional RNNs and LSTMs (Long Short-Term Memory networks). GRUs are similar to LSTMs but with a simpler structure, using fewer gates and parameters while maintaining comparable performance for many sequential tasks.

### Key Features of GRU
The GRU architecture includes two main gates: 
`the reset gate` and `the update gate`. These gates control the flow of information and help the network maintain relevant information over time while forgetting irrelevant data.

- Update Gate (`𝑧𝑡`):
The update gate decides how much of the previous hidden state should be retained.
It ranges from 0 to 1, where 0 means completely forget the previous state, and 1 means completely retain the previous state.
The update gate is computed using the sigmoid function:
    `z𝑡 = 𝜎(𝑊𝑓 ⋅ [ℎ𝑡−1,𝑥𝑡] + 𝑏z)`
- Reset Gate (`𝑟𝑡`):
The reset gate controls how much of the previous hidden state should be ignored when calculating the candidate hidden state.
It is computed similarly to the update gate:
`r𝑡 = 𝜎(𝑊𝑓 ⋅ [ℎ𝑡−1,𝑥𝑡] + 𝑏r)`

- The `candidate hidden state` is a new potential memory influenced by the reset gate.
- The `final hidden state` is a weighted combination of the previous hidden state and the candidate state, with the update gate determining how much of the new information to retain.

### Advantages of GRU

- Simpler Architecture: Fewer parameters than LSTMs, which makes them computationally more efficient and easier to train.
- Good Performance: Despite their simplicity, GRUs achieve performance comparable to LSTMs on many tasks, making them a popular choice for sequence modeling.
- Faster Training: Due to fewer parameters, GRUs are faster to train compared to LSTMs, especially on large datasets.

------------------------------------------------------------------------------------------------------------------------------------------

# `UNIT - 4`

# 1. What is Convolution, stride, padding in CNN?
- **Convolution:** is a `mathematical operation` that involves `sliding a small matrix`, called a filter or kernel, over an input image or feature map. Each position of the filter computes a `weighted sum`, which generates a new matrix called a feature map or activation map. This feature map captures important spatial features from the input, such as edges, textures, and shapes.

- **Stride:** refers to the number of pixels the filter moves after each operation. Stride influences the size of the output feature map.

- **Padding:** is the process of adding extra pixels around the border of an image or input matrix. Padding helps control the output's spatial dimensions and has two main purposes:
    - **Maintain Output Size:** Padding can be used to keep the output the same size as the input.
    - **Reduce Information Loss:** By padding the borders, all parts of the input have the opportunity to contribute to the feature map, including the edges.

# 2. Relation Between Input, Output, and Filter Size
The output size of a convolutional layer depends on the input size, the filter size, stride, and padding. For a given 2D input matrix with dimensions `Hin * Win`, the output dimensions `Hout * Wout` can be calculated with the following formulas:

![](image-10.png), where:

- `Hin` and `Win` Height and width of the input.
- K: Size of the filter (kernel) (assuming a square kernel like 3×3).
- P: Padding size (how many pixels are added around the input).
- S: Stride.

# 3. CNN Architecture :
is a specialized type of neural network that is particularly effective for processing `grid-like data`, such as images. CNNs are commonly used for image classification, object detection, and other visual recognition tasks.

![cnnimage](image-11.png)

### Core Layers of a CNN

- **Convolutional Layer:** 
This layer performs the convolution operation. `It applies several filters` (kernels) to the input image, producing multiple feature maps. Each filter is a small matrix that slides over the input image and captures specific patterns (e.g., edges, textures). The output of the convolutional layer, called the `activation map` or feature map, `highlights areas where specific patterns are detected`.

- **Activation Layer:** 
By adding an activation function to the output of the preceding layer, activation layers add `nonlinearity` to the network. it will apply an element-wise activation function to the output of the convolution layer. Some common activation functions are RELU: max(0, x),  Tanh, Leaky RELU, etc.

- **Pooling layer:** 
reduces the `spatial dimensions` of the feature map (height and width) while `retaining important features`. This decreases the computational load and `helps prevent overfitting`. A `2×2` max pooling operation with a `stride of 2` on a `32×32` feature map reduces it to `16×16`.

- **Fully Connected Layer:**
Flattening is used to convert all the resultant 2-Dimensional arrays from
pooled feature maps into a single long continuous linear vector. The
flattened matrix is fed as input to the fully connected layer to classify the
image.
`(These layers are used after a sequence of convolutional and pooling layers to perform classification based on the features extracted.
Each neuron in a fully connected layer is connected to every neuron in the previous layer.)`

- **Output Layer:**
uses an activation function `(usually softmax for multi-class classification or sigmoid for binary classification)` to convert the final outputs into probabilities.

# 4. What is Weight Sharing?
In CNNs, weight sharing means that the `same set of weights` (the filter or kernel) is applied across different regions of the input image. Instead of having `unique weights for every pixel` (as in a fully connected layer), the `same filter is "shared"` across the entire input. This allows the CNN to detect the `same pattern` or feature (e.g., edges, textures) at `different locations` in the image.
#### Advantages:
- Reduced Number of Parameters:
- Lower Memory and Computation Requirements

# 5. FCNN vs CNN 
- **Architecture**
    - **Fully Connected Neural Network (FCNN):**
        - In an `FCNN`, every neuron in one layer is connected to every neuron in the next layer. These layers are called fully connected or dense layers.
        - Each layer contains `independent weights and biases`, meaning a large number of parameters as each neuron must learn its own set of weights for every connection.

    - **Convolutional Neural Network (CNN):**
        - `CNNs` have specialized layers, primarily convolutional layers and pooling layers, followed by fully connected layers at the end.
        - Instead of connecting each neuron to every pixel, CNNs use a `small filter (kernel) that slides across the input`, allowing the network to focus on local spatial patterns.

- **Parameter Efficiency** 
`CNNs` use `weight sharing` in convolutional layers, significantly reducing the number of parameters compared to `FCNNs`, which are more memory-intensive.
- **Best Use Cases**
`FCNNs` are ideal for `tabular data and tasks without spatial structure`. `CNNs` are well-suited for `image, video, and other tasks with spatial patterns`, as they can recognize features regardless of position.
- **Computational Cost and Memory Usage**
`FCNN` Has `high` computational cost and memory usage due to the large number of parameters, especially for high-dimensional inputs like images, as each pixel has a unique weight.
`CNN` Is much more memory-efficient due to weight sharing and smaller filter sizes, which reduces computational cost while maintaining performance, especially for tasks involving spatial patterns.
- **Translation invarience:** CNNs are translation `invariant`, meaning they `can detect features at different positions` within the input, unlike FCNNs.

# 6. Convolution Types:
- Multichannel Convolution: Used for images with multiple channels (like RGB). Each filter matches the image's depth (e.g., 3 for RGB), applies separately to each channel, and then sums the results to produce one feature map.

- 2D Convolution: Most common for images, sliding a 2D filter over the height and width of the image to capture spatial features. Typically used for image processing where data is in two dimensions.

- 3D Convolution: Extends 2D convolution to include depth, allowing the filter to slide over height, width, and depth. It’s useful for video data (where depth could represent time) or 3D medical scans, capturing patterns across all three dimensions.