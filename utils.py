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

def get_combinations(node_count: int, min_layer_node_count: int = 1):
    r""" Returns an iterator iterating over all of the possible layer node
    count combinations, respecting the minimum layer node count.

    Each combination will be returned as a list of integers indicating the
    node count at each layer.

    Starts with all combinations of nodes for the current layer count before
    increasing it. After layer count increase restarts with all extra nodes
    moved to the first layer, each time moving the first movable node from the
    left to the nearest layer on the right.

    Example for node_count=4, min_layer_node_count=1:
    [4]
    [3, 1]
    [2, 2]
    [1, 3]
    [2, 1, 1]
    [1, 2, 1]
    [1, 1, 2]
    [1, 1, 1, 1]

    """

    layer_sizes = [node_count]
    while True:

        # Return current layer node counts
        yield layer_sizes
        
        # Move a node to other layer
        for index in range(len(layer_sizes)):

            # if the first movable node exists on the last layer
            if index == len(layer_sizes) - 1:
                
                # Increase layer count,
                # distribute at least one node to each layer
                # and put all of the remaining nodes in the first layer.

                # if node count already matches the layer count then exit
                if (len(layer_sizes) >= node_count): 
                    return

                layer_sizes = [min_layer_node_count] * (len(layer_sizes) + 1)

                layer_sizes[0] = (min_layer_node_count + node_count
                                  - len(layer_sizes))

                break # exit node moving

            # else if current layer has more than one node
            elif layer_sizes[index] > min_layer_node_count: 

                # Move the node to the next nearest layer

                layer_sizes[index] = layer_sizes[index] - 1
                layer_sizes[index + 1] = layer_sizes[index + 1] + 1

                break
