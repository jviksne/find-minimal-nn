# Copyright © 2022 Janis Viksne
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
from typing import List, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

# Indicates whether a device for PyTorch calculations has been chosen
device:Literal['cpu','cuda'] = None

def init_device(prefer_device:str = None):
    global device
    
    if device != None:
        return device

    if prefer_device != None:
        device = prefer_device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {device} device")
    return device

class BasicNet(nn.Module):
    r"""A neural network with 
    :math:`y = xA^T + b`
    """

    def __init__(self,
                 in_size:int,
                 hidden_layer_sizes: List[int],
                 out_size: int,
                 learn_rate: float = 1e-3):

        super(BasicNet, self).__init__()

        init_device()

        self.hidden_layer_sizes = hidden_layer_sizes # save to be able to print out

        # Init the layer stack
        self.linear_relu_stack = nn.Sequential()

        # Append layers        
        prev_layer_size = in_size

        for layer_size in hidden_layer_sizes:
            self.linear_relu_stack.append(nn.Linear(
                    in_features=prev_layer_size,
                    out_features=layer_size,
                    bias=True,
                    device=device))
            self.linear_relu_stack.append(nn.Sigmoid())  # Sigmoid activation function - 1 / (1 + torch.exp(-f(x)))
            prev_layer_size = layer_size

        self.linear_relu_stack.append(nn.Linear(in_features=prev_layer_size, out_features=out_size, bias=True, device=device))
        self.linear_relu_stack.append(nn.Sigmoid())

        # print(self.linear_relu_stack) # to print out the layout

        # Init loss function: -(y * log(f(x)) + (1 - y) * log(1 - f(x))) with 100 as the maximum value
        self.loss_fn = nn.BCELoss(reduction='mean')

        # Use gradient descent
        self.optimizer = torch.optim.SGD(self.linear_relu_stack.parameters(), lr=learn_rate)

        self.calculation_count = 0 # increased upon each backpropogation, determines when to print status update

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def train_epoch(self, dataloader: DataLoader, epoch: int):

        global device

        self.train() # enter training mode

        for (X, y) in dataloader:

            #X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad() # clear old gradients
            loss.backward()  # calculate new gradients for all parameters (stored as attributes of parameters)
            self.optimizer.step() # do the update

            self.calculation_count = self.calculation_count + 1
            if self.calculation_count % 100 == 0:
                print (f'Layout {self.hidden_layer_sizes}, epoch {epoch}, loss {loss}', end='\r')

    def is_correct_for_all(self, dataset: Dataset):
        for i in range(len(dataset)): # Dataset may not be iterable over values
            data = dataset[i]
            pred = torch.round(self(data[0]))
            if not torch.equal(pred, data[1]):
                return (data[0], data[1], pred)
        return True

    def print_random_samples(self, dataset: Dataset, count: int = -1):
        if count < 0:
            samples = range(0, len(dataset))
        else:
            samples = random.sample(range(0, len(dataset)), count)
        for sample_index, dataset_index in enumerate(samples):
            X, _ = dataset[dataset_index]
            print(f"---Sample #{sample_index}---\n{X.detach().to('cpu').numpy()}\n{self(X).detach().to('cpu').numpy()}\n")

    def print_weights(self, print_header: bool = True):
        if (print_header):
            print("---Weights:---")
        for name, param in self.linear_relu_stack.named_parameters():
            print(f"{name} {param.detach().to('cpu').numpy()}")

    def get_weights(self, index:int):
        index = index * 2
        for curr_index, param in enumerate(self.linear_relu_stack.parameters()):
            if curr_index == index:
                return param
    
    def get_bias(self, index:int):
        index = index * 2 + 1
        for curr_index, param in enumerate(self.linear_relu_stack.parameters()):
            if curr_index == index:
                return param

    def manual_gd_calc(self, learn_rate: float, x:torch.Tensor, y:torch.Tensor):

        # Feedforward

        node_values = [x.detach().clone()]
        reg_weights = []
        bias_weights = []

        # Loop through regular and bias node weights (which are present in sequental
        # order one by one with regular weights coming first and bias weights afterwards).
        for index, param in enumerate(self.linear_relu_stack.parameters()):
            if index % 2 == 1:
                reg_weights.append(param.detach().clone())
            else:
                bias_weights.append(param.detach().clone())

                # calculate input for activation function from previous nodes and incoming weights
                inp = reg_weights[-1].matmul(node_values[-1]) + bias_weights[-1]
                
                # current layer node calculation via Sigmoid activation function
                node_values.append(1 / (1 + torch.exp(-inp)))

        print(f'Manual prediction: {node_values[-1]}')   # predictions are the node values of the last layer
        
        cost = -(y*torch.log(node_values[-1]) + (1 - y)*torch.log(1 - node_values[-1])) # regularization not implemented: + regularisation ((lambda / 2m) * sum(weights ^ 2))

        print(f'Manual cost: {cost}')

        # Backpropogation (TODO: review whether correct and do not skip bias)

        new_weights = []

        right_nodes = node_values.pop()        
        right_delta = right_nodes - y
        print(f'Delta out: {right_delta}')
        while True:
            weights = reg_weights.pop()
            if not weights:
                break
            weights = weights - learn_rate * right_delta
            left_nodes = node_values.pop()
            g_prime = left_nodes * (1 - left_nodes)
            delta = torch.transpose(weights, 0, 1).matmul(right_delta) * g_prime
            print(f'Delta hidden layer: {delta}')
            new_weights.append((weights.transpose(0, 1) - learn_rate * delta).transpose(0, 1))

        for index, weights in enumerate(new_weights):
            print(f'New weights #{index}:\n{weights}')

