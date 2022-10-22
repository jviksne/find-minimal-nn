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

# To setup and run:
# pip3 install torch
# pip3 install --upgrade numpy
# python main.py

# A very basic Neural Network.
# No splitting into training and test data sets, a NN is considered to be found if it produces a valid output for all inputs.
# No regularization.

from utils import get_combinations
from basicnet import BasicNet, init_device
from logicaldata import LogicalDataset
from comparisondata import NumberComparisonDataset
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, RandomSampler

# Constants
MIN_LAYER_NODE_COUNT = 1 # Minimal neural network layer node count
DEFAULT_LEARN_RATE = 1e-2 # Learning rate (1e-2 = 0.01)
MAX_EPOCHS = 50000 # Max epochs
SECTION_SEPARATOR = "-" * 20
DEFAULT_BATCH_SIZE = 1 # 1: stochastic,
                       # <= 0: batch,
                       # >1 - mini-batch or batch
EVALUATE_AFTER_EPOCHS = 100
PRINT_WEIGHTS = True
PRINT_SAMPLES = True
MAX_NODE_COUNT = 100
PREFER_DEVICE = None # "cpu", "cuda", None
                     # if None then "cuda" will be used if available


def print_section(title: str):
    print(SECTION_SEPARATOR+"\n"+title+"\n"+SECTION_SEPARATOR)


def look_for_minimal_nn(dataset, # no Dataset type hint because using custom functions
                        max_node_count:int = MAX_NODE_COUNT,
                        batch_size:int = DEFAULT_BATCH_SIZE,
                        learn_rate:float = DEFAULT_LEARN_RATE,
                        train_size:int = None,
                        print_sample_count:int = -1):

    if batch_size <= 0:
        batch_size = len(dataset)

    if train_size != None:
        sampler = RandomSampler(data_source=dataset,
                                num_samples=train_size,
                                replacement=True)
    else:
        sampler = None
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            sampler=sampler)

    node_count_found = False

    # loop through node counts starting from minimal node count
    for node_count in range(MIN_LAYER_NODE_COUNT, max_node_count):
        
        # loop through all possible layer configurations for the
        # current node count
        for hidden_layer_sizes in get_combinations(
                                node_count=node_count,
                                min_layer_node_count=MIN_LAYER_NODE_COUNT):

            # build a new NN passing the layer node counts
            nn = BasicNet(in_size=dataset.input_size(),
                        hidden_layer_sizes=hidden_layer_sizes,
                        out_size=dataset.output_size(),
                        learn_rate=learn_rate)

            # do the training
            for i in range(MAX_EPOCHS):

                nn.train_epoch(dataloader, i)

                # check once in a while if all data doesn't already
                # return correct values
                if ((i == MAX_EPOCHS - 1 or i % EVALUATE_AFTER_EPOCHS == 0)
                    and nn.is_correct_for_all(dataset) == True): 

                    print() # print newline after train_epoch because it
                            # keeps printing status on one and the same line 

                    # if correct values are returned for all data
                    # then do not proceed with larger node counts
                    # (but keep checking other configurations with
                    # current count)
                    node_count_found = True 

                    print(f'Works with {node_count} hidden layer node(s) '
                          f'and layout: {hidden_layer_sizes}\n')

                    if PRINT_WEIGHTS:
                        nn.print_weights()
                        print()

                    if PRINT_SAMPLES:
                        nn.print_random_samples(dataset, print_sample_count)

                    break

        if node_count_found:
            break # do not check larger node counts

    if not node_count_found:
        # print newline after train_epoch since it was not printed upon
        # converge
        print() 


device = init_device(PREFER_DEVICE)

# Look for the minimal node count that is capable to solve boolean OR
print_section("Looking for a minimal NN that solves binary 'OR'\n"
              " input: [a, b], output: [a OR b]")
look_for_minimal_nn(LogicalDataset('or', device))

# Look for the minimal node count that is capable to solve boolean AND
print_section("Looking for a minimal NN that solves binary 'AND'\n"
              " input: [a, b], output: [a AND b]")
look_for_minimal_nn(LogicalDataset('and', device))

# Look for the minimal node count that is capable to solve boolean AND
print_section("Looking for a minimal NN that solves binary 'XOR'\n"
              " input: [a, b], output: [a XOR b]")
look_for_minimal_nn(LogicalDataset('xor', device))

# Look for the minimal node count that is capable to compare numbers
# of certain sizes
for max_val, train_size, batch_size, learn_rate, do_mean_norm in [
        [1, None, DEFAULT_BATCH_SIZE, DEFAULT_LEARN_RATE, False],
        [10, None, DEFAULT_BATCH_SIZE, DEFAULT_LEARN_RATE, True],
        [99, 1000, 100, 1e-2, True],
        [999, 100000, 100, 1e-5, True]
    ]:
    print_section(f"Looking for a minimal NN that can compare numbers "
                  f"<={max_val}\n input: [a, b], output: [a < b, a > b] "
                  f"where 0 <= a,b <= {max_val}")
    look_for_minimal_nn(dataset=NumberComparisonDataset(max_val,
                                                        device,
                                                        do_mean_norm=do_mean_norm),
                        print_sample_count=4,
                        train_size=train_size,
                        batch_size=batch_size,
                        learn_rate=learn_rate)

