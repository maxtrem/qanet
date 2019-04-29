
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.conv import Initialized_Conv1d, DepthwiseSeparableCNN
from modules.attn import MultiHeadAttnBL as MultiHeadAttn
from modules.pos_enc import PositionalEncoding


class ResidualBlock(nn.Module):
    """
    ResidualBlock implements the function in form of `f(layernorm(x)) + x`. 
    Dropout and activation function can be set as well.
    """
    def __init__(self, layer, shape, norm=None, droprate=0.0, activation=nn.ReLU, shared_weight=False, shared_norm=False):
        """
        # Arguments
            layer:      (instance of nn.Module with forward() implemented), layer represents function f
            shape:      (tuple of int) shape of the last two dimensions of input
            activation: sets the activation function for the network, valid options can be seen in
                        the "torch.nn.modules.activation" module. Use None for no activation
            droprate:   (float) sets dropout rate for dropout
        """

        super().__init__()
        
        if shared_norm:
            # using __dict__.update prevents the layer module from being registered in parameters
            self.__dict__.update({'norm': shared_norm})
        else:
            self.norm  = nn.LayerNorm(shape)
        if shared_weight:
            # prevents the layer module from being registered in parameters
            self.__dict__.update({'layer': layer})
        else:
            self.layer = layer
        self.activation_ = activation() if activation else None
        self.dropout     = nn.Dropout(p=droprate)
        
    def activation(self, x):
        return self.activation_(x) if self.activation_ else x
        
    def forward(self, *args, **kwargs):
        if 'x' in kwargs:
            x = kwargs['x']
            del kwargs['x']
        else:
            x = args[0]
            args = args[1:]
        residual_x = x
        x = self.norm(x)
        x = self.layer(x, *args, **kwargs)
        x = self.activation(x) + residual_x
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    """
    Each EncoderBlock consists of the following layers:
    
    (1): Positional Encoding Layer
    (2): Stacked Convolutional Layers
    (3): Multi-Head-Self-Attention Layer
    (4): Feed Forward Layer
    
    Each of these layers is placed inside a residual block.
    """
    def __init__(self, d_model=128, seq_limit=25, kernel_size=7, num_conv_layers=4, droprate=0.0, shared_weight=False, shared_norm=False):
        """
        # Arguments
            d_model:     (int) dimensionality of the model
            seq_limit:   (int) length of the padded / truncated sequences
            kernel_size: (int or tuple) size of the convolving kernel
            num_conv_layers: (int) number of stacked convolutional layers used in the block
            droprate:    (float) sets dropout rate for dropout
        """
        super().__init__()
        # handing over shape to init LayerNorm layer
        shape = d_model, seq_limit
        #self.positional_encoding_layer = PositionalEncoding(d_model, seq_limit, droprate=0.0)
        self.positional_encoding_layer = PositionalEncoding(d_model, seq_limit)

        shared_keys = {'conv_layers', 'mh_attn', 'ffnet'}

        if shared_weight:
            missing = set.difference(shared_keys, set(shared_weight.keys()))
            assert missing == set(), f'Missing modules {missing}'
            self.main_layers = shared_weight
            shared = True


        else:
            conv_layers = [DepthwiseSeparableCNN(d_model, d_model, kernel_size=kernel_size, activation=nn.ReLU) for _ in range(num_conv_layers)]
            mh_attn = MultiHeadAttn(d_model=d_model, heads=8, droprate=droprate)#, proj_type=2)
            ffnet = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)

            self.main_layers = {'conv_layers': conv_layers,
                                'mh_attn': mh_attn,
                                'ffnet': ffnet}
            shared = False

        if shared_norm:
            self.shared_norm = nn.LayerNorm(shape)
        else:
            self.shared_norm = False
        # ToDO: Try shared norm layer
        stacked_CNN = [ResidualBlock(conv_layer, shape=shape, shared_weight=shared, shared_norm=self.shared_norm) for conv_layer in self.main_layers['conv_layers']]
        self.conv_blocks = nn.Sequential(*stacked_CNN)
        
        self.self_attn_block = ResidualBlock(self.main_layers['mh_attn'], shape=shape, activation=None, droprate=droprate, ##RELU??
                                             shared_weight=shared, shared_norm=self.shared_norm)
        
        self.feed_forward = ResidualBlock(self.main_layers['ffnet'], shape=shape, activation=None, droprate=droprate, 
                                          shared_weight=shared, shared_norm=self.shared_norm)

        
        
    def forward(self, x, mask=None):
        x = self.positional_encoding_layer(x)
        x = self.conv_blocks(x)
        x = self.self_attn_block(x, mask=mask)
        x = self.feed_forward(x)
        return x