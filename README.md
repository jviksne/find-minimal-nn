# find-minimal-nn
A toy project for trying out PyTorch. Looks for a minimal neural network capable to perform "OR", "AND", "XOR" boolean operations and less than "<" and greater than ">" comparisons.

The script tries out all possible hidden layer node layout combinations increasing the node count until the neural network produces valid outputs for all the inputs (the current target data sets are of limited size). If a certain node count produces a valid input, the process continues to check all other node layouts for the same count, if any remain to be checked.

The structure of the generated neural networks match the basic neural networks taught by Andrew Ng in the excellent Coursera Machine Learning course of Stanford University (more details below), apart from the regularization (penalization of large weight values that increase the fluctuation and possibly causing overfitting) which has not been added due to the limited all possible input set and overfitting not being a concern.

Note that the neural network may get stuck in local optima or may not converge and the results should not be considered to prove, without additional analysis, that a certain network layout is not capable of producing valid outputs.
 
# Setup and run

Install PyTorch by following the instructions from https://pytorch.org/.

Download and extract the files from this repository to some directory and run:
```bash
python main.py
```

# Configuration

* Default learning rate is 0.01.
* Batch size

# Implementation details

* Each node of the previous layer is connected to each node of the following layer.
* The input for the activation function of the following layer is calculated by summing the multiplication of incoming weights with incoming node values.
* The activation function for each node is:
* Loss function is:
* Learning algorithm is gradient descend, which, depending on the configured sample size can behave as a stochastic gradient descent (if batch_sample_size is set to 1), mini-batch (if batch_sample_size is set to less than the dataset size) or batch (if batch_sample_size matches the dataset size).

When iterating over the all possible combinations of a certain node count the algorithm arranges the nodes the following way:


# Results

AND and OR operations work with a single node hidden layer (it would work with no hidden layers at all and weights going straight to the output node). Sample output:

XOR requires two nodes to be arranged in a single hidden layer where nodes cancel each other when weights go to the output layer if both of them are activated. Sample output:

Comparison of two integers (with neural network producing valid inequality operation - ">" and "<" - results in two output nodes and both being 0 indicating an equality) depends on the maximum size of the integer:

# Issues

Need to figure out why GPU processing on my hardware seems slower than the CPU one. Possibly it is due to the nature of the data sets, but it may also be some code or my environment configuration issue.
