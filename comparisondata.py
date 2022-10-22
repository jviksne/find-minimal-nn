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

import torch
from torch.utils.data import Dataset

class NumberComparisonDataset(Dataset):
    
    def __init__(self,
                 max: int,
                 device: str = None,
                 do_mean_norm: bool = True):
        super(NumberComparisonDataset).__init__()
        self.up_to = max + 1
        self.device = device
        self.do_mean_normalization = do_mean_norm
        self.mean = sum(range(self.up_to)) / self.up_to

    def __len__(self):
        return self.up_to * self.up_to

    def input_size(self):
        return 2 # the numbers to compare

    def output_size(self):
        return 2 # is less than, is greater than

    def mean_normal(self, num):
        return (num - self.mean) / float(self.up_to)

    def __getitem__(self, index):

        [[quotient, modulo], _] = self.get_input(index, False)

        if quotient < modulo:
            is_smaller = 1.
        else:
            is_smaller = 0.
        if quotient > modulo:
            is_greater = 1.
        else:
            is_greater = 0.
        return (torch.tensor(data = [quotient, modulo],
                             device = self.device),
                torch.tensor(data = [is_smaller, is_greater],
                             device = self.device))

    def get_input(self,
                                       index,
                                       train_data_as_tensor: bool):
        quotient, modulo = divmod(index, self.up_to)

        quotient, modulo = float(quotient), float(modulo)

        if self.do_mean_normalization:
            quotient_norm = self.mean_normal(quotient)
            modulo_norm = self.mean_normal(modulo)
        else:
            quotient_norm = quotient
            modulo_norm = modulo

        train_data = [quotient_norm, modulo_norm]
        if train_data_as_tensor:
            train_data = torch.tensor(data=train_data,
                                      device=self.device)

        return [train_data, [quotient, modulo]]


    # To implement an IterableDataset instead of Dataset, add:
    # def __iter__(self):
    #     for index in range(len(self)):
    #         yield self[index]
        