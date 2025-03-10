import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from KANConv import KAN_Convolution
from KANLinear import KANLinear


class KAN_EEG(nn.Module):

    def __init__(self):
        super(KAN_EEG, self).__init__()

        # Layer1
        self.kanlayer11 = nn.Sequential(
            KAN_Convolution(30, 15, 3, padding=1),
            nn.BatchNorm1d(30, False)
        )
        self.kanlayer12 = nn.Sequential(
            KAN_Convolution(15, 7, 3, padding=1),
            nn.BatchNorm1d(15, False)
        )
        self.kanlayer13 = nn.Sequential(
            KAN_Convolution(7, 3, 3, padding=1),
            nn.BatchNorm1d(7, False)
        )
        self.pooling1 = nn.MaxPool1d(2, stride=2)
        self.pooling2 = nn.MaxPool1d(2, stride=2)
        self.pooling3 = nn.MaxPool1d(2, stride=2)

        # self.fc = KANLinear(, 2)


    def forward(self, x):
        # x=[batch, channels, timepoint]
        print('enter')
        x = self.kanlayer11(x)
        x = self.pooling1(x)
        print(x.shape)

        x = self.kanlayer12(x)
        x = self.pooling2(x)
        print(x.shape)

        x = self.kanlayer13(x)
        x = self.pooling3(x)
        print(x.shape)

        x = nn.Flatten(x)
        print(x.shape)
        # x = self.fc(x)

        return x

if __name__ == '__main__':
    data = torch.Tensor(np.random.rand(1, 30, 256))
    # model = KAN_Convolution(30, 15, 3, padding=1)
    model = nn.Conv1d(30, 15, 3, padding=1)
    result = model(data)

    print(result.shape)
    print(result)




