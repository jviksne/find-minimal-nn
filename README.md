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
* The input for the activation function of the following layer is calculated by summing the multiplication of incoming weights with incoming node values: $y = xA^T + b$
* The activation function for each node is: $S(x)={\frac {1}{1+e^{-x}}}$
* The loss function is: $J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]$
* The learning algorithm is gradient descend, which, depending on the configured sample size can behave as a stochastic gradient descent (if batch_sample_size is set to 1), mini-batch (if batch_sample_size is set to less than the dataset size) or batch (if batch_sample_size matches the dataset size).
* The comparsion function input data for input sets that contain values greater than 1 are preprocessed by applying mean normalization: $x_i := \dfrac{x_i - \mu_i}{s_i}$
* Learning rate: 0.01

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

OR and AND operations work with a single node hidden layer (it would work with no hidden layers at all and weights going straight to the output node).

* Sample output for OR:

```
0.weight [[2.2405872 2.2749467]]
0.bias [-1.223995]
2.weight [[3.4729319]]
2.bias [-0.85417444]
```

* Sample output for AND:

```
0.weight [[-1.9562051 -1.9832208]]
0.bias [2.1289062]
2.weight [[-4.1473875]]
2.bias [0.6603181]
```

XOR requires two nodes to be arranged in a single hidden layer where nodes cancel each other when weights go to the output layer if both of them are activated. Sample output:

```
0.weight [[4.058795  4.177402 ]
 [1.2133992 1.3914024]]
0.bias [-1.0793928 -1.5911556]
2.weight [[ 4.158029  -3.0204496]]
2.bias [-1.9593607]
```

Comparison of two integers (with neural network producing valid inequality operation - ">" and "<" - results in two output nodes and both being 0 indicating an equality) works with a single node hidden layer.

* Sample output for comparing 0 and 1:

```
0.weight [[-3.0167425  2.9965193]]
0.bias [-0.35470104]
2.weight [[ 4.000025] [-3.651333]]
2.bias [-3.0536509   0.12493792]
```

* Sample output for comparing integers from 0 to 10  (note that for this and larger number comparisons the mean normalization is applied to the inputs):

```
0.weight [[-4.450456   4.5212965]]
0.bias [-0.20523731]
2.weight [[ 4.483545] [-4.940106]]
2.bias [-2.2536952  1.9412361]
```

* Sample output for comparing integers from 0 to 99:

```
0.weight [[ 5.658082 -5.658709]]
0.bias [-0.17585063]
2.weight [[-5.9274735] [ 6.230655 ]]
2.bias [ 2.631637  -2.8514955]
```

* Sample output for for comparing integers from 0 to 999:

```
0.weight [[ 11.733412 -11.735888]]
0.bias [-0.13835377]
2.weight [[-12.713495]
 [ 12.741911]]
2.bias [ 5.9011207 -5.947039 ]
```

# Issues

Need to figure out why GPU processing on my hardware seems slower than the CPU one. Possibly it is due to the nature of the data sets, but it may also be some code or my environment configuration issue.
