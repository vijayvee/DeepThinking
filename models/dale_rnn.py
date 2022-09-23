import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from models.dale_rnn_layer import *

#Dale-RNN
class DaleRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.kernal_size = 5
        self.padding = 2
        self.inplanes = 32
        self.width = 2  # hard coded, following ResidualNetworkSegment style of model
        self.inplanes = self.inplanes * self.width

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        
        self.rnn = DaleRNNLayer(self.inplanes, self.inplanes, 
                                3, 5, 3, timesteps=15)

        self.conv2 = nn.Conv2d(self.inplanes, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

    def forward(self, x):  
        x = F.relu(self.bn1(self.conv1(x)))
        print(x.shape)
        x = self.rnn(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

def dalernn(num_outputs, depth, width, dataset):
    return DaleRNN()