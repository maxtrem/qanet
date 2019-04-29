
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.conv import Initialized_Conv1d

class FeedForward(nn.Module):
    """Implements a simple feedfoward layer, which can be switched between modes CNN and Linear"""
    def __init__(self, in_features, out_features, droprate=0.0, activation=nn.ReLU, bias=True, use_cnn=True):
        """
        # Arguments
            in_features:  (int) number of input features
            out_features: (int) number of output features
            droprate:     (float) sets dropout rate for dropout
            activation:   sets the activation function for the network, valid options can be seen in
                          the "torch.nn.modules.activation" module. Use None for no activation
            bias:         (bool) activates bias if set True
            use_cnn:      (bool) if True a convolutional layer is used instead of a linear layer
        """
        super().__init__()
        self.bias    = bias
        self.use_cnn = use_cnn
        if use_cnn:
            self.ff_layer = Initialized_Conv1d(in_features, out_features, relu=False, bias=True)
            
        else:
            self.ff_layer = nn.Linear(in_features, out_features, bias)
            if bias:
                self.ff_layer.bias.data = self.ff_layer.bias.view(-1, 1).contiguous()

        self.activation_ = activation() if activation else None
        self.dropout     = nn.Dropout(p=droprate)
            
    def activation(self, x):
        return self.activation_(x) if self.activation_ else x
    
    def forward(self, x):
        if self.use_cnn:
            return self.forward_cnn(x)
        else:
            return self.forward_lin(x)
        
    def forward_cnn(self, x):
        x = self.ff_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
        
    def forward_lin(self, x):
        x = self.ff_layer.weight @ x
        if self.bias:
            x += torch.jit._unwrap_optional(self.ff_layer.bias)
        x = self.dropout(self.activation(x))
        return x

