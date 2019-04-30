
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.helpers import Activation
from modules.conv import Initialized_Conv1d, DepthwiseSeparableCNN
from modules.attn import MultiHeadAttn as MultiHeadAttn
from modules.pos_enc import PositionalEncoding


class ResidualBlock(nn.Module):
    """
    ResidualBlock implements the function in form of `f(layernorm(x)) + x`. 
    Dropout and activation function can be set as well.
    """
    def __init__(self, layer, d_model, droprate=0.0, activation=None):
        """
        # Arguments
            layer:      (instance of nn.Module with forward() implemented), layer represents function f
            shape:      (tuple of int) shape of the last two dimensions of input
            activation: sets the activation function for the network, valid options can be seen in
                        the "torch.nn.modules.activation" module. Use None for no activation
            droprate:   (float) sets dropout rate for dropout
        """

        super().__init__()
        

        self.norm  = nn.LayerNorm(d_model)
        self.layer = layer
        self.activation  = Activation(activation)
        self.dropout     = nn.Dropout(p=droprate)
        
    def forward(self, *args, **kwargs):
        if 'x' in kwargs:
            x = kwargs['x']
            del kwargs['x']
        else:
            x = args[0]
            args = args[1:]
        residual_x = x
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.layer(x, *args, **kwargs)
        x = self.activation(x) 
        x = self.dropout(x)
        return x + residual_x

class EncoderBlock(nn.Module):
    """
    Each EncoderBlock consists of the following layers:
    
    (1): Positional Encoding Layer
    (2): Stacked Convolutional Layers
    (3): Multi-Head-Self-Attention Layer
    (4): Feed Forward Layer
    
    Each of these layers is placed inside a residual block.
    """
    def __init__(self, d_model=128, seq_limit=400, kernel_size=7, num_conv_layers=4, droprate=0.0):
        """
        # Arguments
            d_model:     (int) dimensionality of the model
            seq_limit:   (int) length of the padded / truncated sequences
            kernel_size: (int or tuple) size of the convolving kernel
            num_conv_layers: (int) number of stacked convolutional layers used in the block
            droprate:    (float) sets dropout rate for dropout
        """
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, seq_limit)

        conv_layers = [DepthwiseSeparableCNN(d_model, d_model, kernel_size=kernel_size) for _ in range(num_conv_layers)]
        stacked_CNN = [ResidualBlock(cl, d_model, droprate) for cl in conv_layers]
        self.conv_blocks = nn.Sequential(*stacked_CNN)
        
        mh_attn = MultiHeadAttn(d_model=d_model, heads=8, droprate=droprate)#, proj_type=2)
        self.self_attn_block = ResidualBlock(mh_attn, d_model, droprate)
        l1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        #l1 = RegularConv(d_model, d_model, activation=nn.ReLU(), bias=True)

        #l2 = Initialized_Conv1d(d_model, d_model, relu=False, bias=True)
        #l12= nn.Sequential(l1, l2)
        self.feed_forward = ResidualBlock(l1, d_model, droprate)

        
        
    def forward(self, x, mask=None):
        x = self.pos_encoder(x)
        x = self.conv_blocks(x)
        x = self.self_attn_block(x, mask=mask)
        x = self.feed_forward(x)
        return x
    