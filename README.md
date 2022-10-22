# find-minimal-nn
A toy project for trying out PyTorch. Looks for a minimal neural network capable to perform "OR", "AND", "XOR" boolean operations and less than "<" and greater than ">" comparisons.

The script tries out all possible hidden layer node layout combinations increasing the node count until the neural network produces valid outputs for all the inputs (the current target data sets are of limited size). If a certain node count produces a valid input, the process continues to check all other node layouts for the same count, if any remain to be checked.

The structure of the generated neural networks match the basic neural networks taught by Andrew Ng in the excellent Coursera Machine Learning course of Stanford University (more details below), apart from the regularization (penalization of large weight values that increase the fluctuation and possibly causing overfitting) which has not been added due to the limited all possible input set and overfitting not being a concern.

Note that the neural network may get stuck in local optima or may not converge and the results should not be considered to prove, without additional analysis, that a certain network layout is not capable of producing valid outputs.
 
# Setup and run

Install PyTorch by following the instructions at https://pytorch.org/.

Download and extract the files from this repository to some directory and run:
```bash
python main.py
```

# Implementation details

* Each node of the previous layer is connected to each node of the following layer.
* The input for the activation function of the following layer is calculated by summing the multiplication of incoming weights with incoming node values.
* The activation function for each node is: $S(x)={\frac {1}{1+e^{-x}}}$
* The loss function is: $J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]$
* The learning algorithm is gradient descend, which, depending on the configured sample size can behave as a stochastic gradient descent (if batch_sample_size is set to 1), mini-batch (if batch_sample_size is set to less than the dataset size) or batch (if batch_sample_size matches the dataset size).
* The comparsion function input data are preprocessed by applying mean normalization: $x_i := \dfrac{x_i - \mu_i}{s_i}$


When iterating over all of the possible combinations of a certain node count the algorithm begins with all combinations of nodes for the current layer count before increasing it. After layer count increases it restarts with all extra nodes moved to the first layer, each time moving the first movable node from the left to the nearest layer on the right.

For example, if node count is 4 and minimal layer node count is set to 1 then the sequence of the combinations tried out will be the following.
```
    [4]
    [3, 1]
    [2, 2]
    [1, 3]
    [2, 1, 1]
    [1, 2, 1]
    [1, 1, 2]
    [1, 1, 1, 1]
```

# Results

AND and OR operations work with a single node hidden layer (it would work with no hidden layers at all and weights going straight to the output node). Sample output:

XOR requires two nodes to be arranged in a single hidden layer where nodes cancel each other when weights go to the output layer if both of them are activated. Sample output:

Comparison of two integers (with neural network producing valid inequality operation - ">" and "<" - results in two output nodes and both being 0 indicating an equality) works with a single node hidden layer:

* weights for comparing 0 and 1:

```
0.weight [[ 2.393474 -2.827444]]
0.bias [0.27705863]
2.weight [[-3.7296197] [ 2.5218446]]
2.bias [ 0.40044788 -2.3560894 ]
```

* weights for comparing integers from 0 to 10:

```
0.weight [[-2.2160113  2.1910453]]
0.bias [0.0598193]
2.weight [[ 4.40381 ] [-4.573212]]
2.bias [-2.3822563  1.6917874]
```

* weights for comparing integers from 0 to 99:

```
0.weight [[-5.195115   5.1957483]]
0.bias [-0.06437927]
2.weight [[  9.759042] [-10.104785]]
2.bias [-5.2375793  4.279991 ]
```

* weights for comparing 0 to 999:

# Issues

Need to figure out why GPU processing on my hardware seems slower than the CPU one. Possibly it is due to the nature of the data sets, but it may also be some code or my environment configuration issue.
