
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.helpers import Activation
from modules.conv import Initialized_Conv1d, DepthwiseSeparableCNN, RegularConv
from modules.attn import MultiHeadAttn as MultiHeadAttn
from modules.pos_enc import PositionalEncoding


class ResidualBlock(nn.Module):
    """
    ResidualBlock implements the function in form of `f(layernorm(x)) + x`. 
    Dropout and activation function can be set as well.
    """
    def __init__(self, layer, d_model, droprate=0.0):
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
        self.droprate = droprate
        
    def forward(self, x, sublayers, mask=None, dropout=True):
        l, L = sublayers
        residual_x = x
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        if dropout:
            x = F.dropout(x, p=self.droprate, training=self.training)
            
        x = (x,) if isinstance(mask, type(None)) else (x, mask)
        x = self.layer(*x)

        x = self.layer_dropout(x, residual_x, droprate=self.droprate * float(l) / L)
        return x, (l+1, L)
    
    def layer_dropout(self, inputs, residual, droprate):
        drop_layer = bool(torch.rand(1) < droprate) and self.training
        return residual if drop_layer else F.dropout(inputs, p=droprate, training=self.training) + residual


class ConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size, droprate=0.1, num_layers=4):
        super().__init__()
        conv_layers = [DepthwiseSeparableCNN(d_model, d_model, kernel_size=kernel_size) for _ in range(num_layers)]
        stacked_CNN = [ResidualBlock(cl, d_model, droprate) for cl in conv_layers]
        self.conv_blocks = nn.Sequential(*stacked_CNN)
        self.droprate = droprate
        
    def forward(self, x, sublayers):
        l, L = sublayers
        for i, block in enumerate(self.conv_blocks):
            x, (l, L) = block(x, (l, L), dropout=(i % 2 == 0))
        return x, (l, L)

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
        self.conv_block = ConvBlock(d_model, kernel_size=kernel_size, droprate=droprate, num_layers=num_conv_layers)
        
        mh_attn = MultiHeadAttn(d_model=d_model, heads=8, droprate=droprate)
        self.self_attn_block = ResidualBlock(mh_attn, d_model, droprate)
        
        ff1 = RegularConv(d_model, d_model, activation=nn.ReLU(), bias=True)
        ff2 = RegularConv(d_model, d_model, activation=None, bias=True)
        ff= nn.Sequential(ff1, ff2)
        self.feed_forward = ResidualBlock(ff, d_model, droprate)
        
    def forward(self, x, sublayers, mask=None):
        l, L = sublayers
        x = self.pos_encoder(x)
        x, sublayers = self.conv_block(x, sublayers)
        x, sublayers = self.self_attn_block(x, sublayers, mask=mask)
        x, sublayers = self.feed_forward(x, sublayers)
        return x, sublayers


class ModelEncoder(nn.Module):
    def __init__(self, d_model=128, seq_limit=400, kernel_size=7, num_conv_layers=4, droprate=0.0, num_blocks=1):
        super().__init__()
        stacked_encoder_blocks     = [EncoderBlock(d_model=d_model, seq_limit=seq_limit, 
                                                   kernel_size=kernel_size, num_conv_layers=num_conv_layers, 
                                                   droprate=droprate) for _ in range(num_blocks)]
        self.stacked_enc_block     = nn.Sequential(*stacked_encoder_blocks)
        self.num_conv_layers = num_conv_layers
        self.num_blocks      = num_blocks

    def forward(self, x, mask=None):
        sublayers = (1, (self.num_conv_layers+2)*self.num_blocks)                          # ToDo: Encoder Stack Wrapper / ModelEncoder?
        for block in self.stacked_enc_block:
            x, sublayers = block(x, sublayers, mask=mask)
        return x



