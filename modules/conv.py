
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.helpers import Activation

# bangliu

class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))

# normal model

class RegularConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, dim=1, activation=None, bias=False):
        super().__init__()

        self.dim = dim
        CNN =  getattr(nn, f'Conv{dim}d')
        
        if isinstance(kernel_size, tuple):
            padding = tuple(map(lambda k: k // 2, kernel_size))
        else:
            padding = kernel_size // 2

        self.conv = CNN(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.activation  = Activation(activation)

        if self.activation:
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity=self.activation.get_str().lower())
        else:
            nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DepthwiseSeparableCNN(nn.Module):
    """
    Implements a depthwise separable convolutional layer as defined by:
    Lukasz Kaiser, Aidan N Gomez, and Francois Chollet. 
    Depthwise separable convolutions for neural machine translation. arXiv preprint arXiv:1706.03059, 2017.
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, dim=1, activation=nn.ReLU(), bias=True):
        """
        # Arguments
            in_channels:   (int) number of input channels
            out_channels:  (int)  number of output channels
            kernel_size:   (int or tuple) size of the convolving kernel
            dim:           (int:[1, 2, 3]) type of convolution i.e. dim=2 results in using nn.Conv2d
            bias:          (bool) controlls usage of bias for convolutional layers
        """
        super().__init__()
        CNN =  getattr(nn, f'Conv{dim}d')
        self.dim = dim
        self.depthwise_cnn = CNN(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=chin, bias=bias)
        self.pointwise_cnn = CNN(in_channels, out_channels, kernel_size=1, bias=bias)
        self.activation  = Activation(activation)

    def forward(self, x):
        x = self.depthwise_cnn(x)
        x = self.pointwise_cnn(x)
        x = self.activation(x)
        return x
