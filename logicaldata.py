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
from typing import Literal

class LogicalDataset(Dataset):
    def __init__(self,
                 op: Literal['or', 'and', 'xor'],
                 device: str = None,
                 do_mean_normalization: bool = False):
        super(LogicalDataset).__init__()
        self.op = op
        self.device = device
        self.enable_mean_normalization(do_mean_normalization)

    def enable_mean_normalization(self, enable: bool = True):
        self.do_mean_normalization = enable

    def __len__(self):
        return 2 ** 2

    def input_size(self):
        return 2 # two boolean operands

    def output_size(self):
        return 1 # true or false

    def __getitem__(self, index):

        [[quotient, modulo], _] = self.get_input(index, False)

        res = 0.

        if self.op == 'or':
                if quotient > 0 or modulo > 0:
                    res = 1.
        elif self.op == 'and':
                if quotient > 0 and modulo > 0:
                    res = 1.
        elif self.op == 'xor':
                if quotient != modulo:
                    res = 1.

        return (torch.tensor(data = [quotient, modulo],
                             device = self.device),
                torch.tensor(data = [res,],
                             device = self.device))

    def get_input(self,
                                       index,
                                       train_data_as_tensor: bool):
        quotient, modulo = divmod(index, 2)

        quotient, modulo = float(quotient), float(modulo)

        if self.do_mean_normalization:
            quotient_norm = quotient - 0.5
            modulo_norm = modulo - 0.5
        else:
            quotient_norm = quotient
            modulo_norm = modulo

        train_data = [quotient_norm, modulo_norm]
        if train_data_as_tensor:
            train_data = torch.tensor(data=train_data,
                                      device=self.device)

        return [train_data, [quotient, modulo]]


#data = LogicalDataset('xor')
#print(len(data))
#for index in range(len(data)):
#   print(data[index])
        