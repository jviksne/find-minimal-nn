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
    def __init__(self, op: Literal['or', 'and', 'xor'], device: str = None):
        super(LogicalDataset).__init__()
        self.op = op
        self.device = device

    def __len__(self):
        return 2 ** 2

    def input_size(self):
        return 2 # two boolean operands

    def output_size(self):
        return 1 # true or false

    def __getitem__(self, index):
        quotient, modulo = divmod(index, 2)

        res = 0.

        match self.op:
            case 'or':
                if quotient == 1 or modulo == 1:
                    res = 1.
            case 'and':
                if quotient == 1 and modulo == 1:
                    res = 1.
            case 'xor':
                if quotient != modulo:
                    res = 1.

        return (torch.tensor(data = [float(quotient), float(modulo)],
                             device = self.device),
                torch.tensor(data = [res,],
                             device = self.device))


#data = LogicalDataset('xor')
#print(len(data))
#for index in range(len(data)):
#   print(data[index])
        