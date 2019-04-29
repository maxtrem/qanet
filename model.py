import torch
import torch.nn as nn
import torch.nn.functional as F

import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def apply_mask(target, mask, eps=-1e30):
    return target * mask + (1 - mask) * (eps)

from modules.pos_enc import PositionalEncoding

from modules.helpers import Activation
###  CNNs

from modules.conv import DepthwiseSeparableCNN, DepthwiseSeparableConv, Initialized_Conv1d, RegularConv

from modules.ffnet import FeedForward

from modules.embedding import InputEmbedding

from modules.attn import SelfAttention as MultiHeadAttn

            
    
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
    def __init__(self, d_model=128, seq_limit=25, kernel_size=7, num_conv_layers=4, droprate=0.0, shared_weight=False, shared_norm=False, pe=None):
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
        if pe == None:
            self.positional_encoding_layer = PositionalEncoding(d_model, seq_limit)
        else:
            self.__dict__.update({'positional_encoding_layer', pe})
        shared_keys = {'conv_layers', 'mh_attn', 'ffnet'}

        if shared_weight:
            missing = set.difference(shared_keys, set(shared_weight.keys()))
            assert missing == set(), f'Missing modules {missing}'
            self.main_layers = shared_weight
            shared = True


        else:
            conv_layers = [DepthwiseSeparableCNN(d_model, d_model, kernel_size=kernel_size, activation=nn.ReLU) for _ in range(num_conv_layers)]
            mh_attn = MultiHeadAttnC(d_model=d_model, heads=8, proj_type=2)
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


class PointerNet(nn.Module):
    """Implements a Pointer Network as defined by:
    Shuohang Wang and Jing Jiang. Machine comprehension using match-lstm and answer pointer. 
    CoRR, abs/1608.07905, 2016. URL http://arxiv.org/abs/1608.07905."""
    def __init__(self, in_features):
        """
        # Arguments
            in_features: int, sets number of input features
        """
        super().__init__()
        self.feedforward = FeedForward(in_features=in_features, out_features=1, bias=False, use_cnn=False, activation=None, droprate=0.0)
        
    def forward(self, x, mask):
        x = self.feedforward(x).squeeze()
        x = apply_mask(x, mask)
        return x

from modules.cqattn import ContextQueryAttention

class QANet(nn.Module):
    """
    Implements QANet as defined by:
    Yu, Adams Wei, et al. "Qanet: Combining local convolution with global self-attention for reading comprehension." 
    arXiv preprint arXiv:1804.09541 (2018).
    """
    def __init__(self, d_model, c_limit, q_limit, word_emb_matrix, char_emb_matrix, droprate=0.0):
        """
        # Arguments
            d_model:     (int) dimensionality of the model
            c_limit:     (int) fixed / maximum length of context sequence
            q_limit:     (int) fixed / maximum length of question sequence
            word_emb_matrix: (numpy.ndarray) embedding weights for tokens
            char_emb_matrix: (numpy.ndarray) embedding weights for characters
        """
        super().__init__()
        
        self.c_limit, self.q_limit = c_limit, q_limit
        
        self.input_embedding_layer = InputEmbedding(word_emb_matrix, char_emb_matrix, d_model=d_model, char_cnn_type=2)
        pe = PositionalEncoding(d_model, c_limit)
        self.positional_encoding_layer = pe
        # ToDo: EncoderBlock shared weights compability for loading weights
        self.context_encoder       = EncoderBlock(d_model=d_model, seq_limit=c_limit, kernel_size=7, droprate=droprate, shared_norm=True, pe=pe)
        self.question_encoder      = EncoderBlock(d_model=d_model, seq_limit=q_limit, kernel_size=7, droprate=droprate, 
                                                  shared_weight=self.context_encoder.main_layers, pe=pe)
        
        self.context_query_attn_layer = ContextQueryAttention(d_model, droprate)
        
        self.CQ_projection         = Initialized_Conv1d(d_model * 4, d_model)
        
        stacked_encoder_blocks     = [EncoderBlock(d_model=d_model, seq_limit=c_limit, kernel_size=5, num_conv_layers=2, droprate=droprate, pe=pe) for _ in range(7)]
        self.stacked_enc_block     = nn.Sequential(*stacked_encoder_blocks)
        
        self.p_start         = PointerNet(2*d_model)
        self.p_end           = PointerNet(2*d_model)

    def forward_stacked_enc_blocks(self, x, mask=None):
        for block in self.stacked_enc_block:
            x = block(x, mask=mask)
        return x
        
    def forward(self, cwids, ccids, qwids, qcids):
        """
        # Arguments
            cwids: (torch.LongTensor) context token ids
            ccids: (torch.LongTensor) context character ids
            qwids: (torch.LongTensor) question token ids
            qcids: (torch.LongTensor) question character ids
            
        # Result
            logits_start: (torch.FloatTensor) logit scores for start of answer in context
            logits_end:   (torch.FloatTensor) logit scores for end of answer in context
        """

        #mask_C = (cwids != 0).float()
        #mask_Q = (qwids != 0).float()

        mask_C = (torch.ones_like(cwids) *
                 0 != cwids).float()
        mask_Q = (torch.ones_like(qwids) *
                 0 != qwids).float()

        C = self.input_embedding_layer(cwids, ccids)
        Q = self.input_embedding_layer(qwids, qcids)
        
        C = self.context_encoder(C, mask_C)
        Q = self.question_encoder(Q, mask_Q)
        
        x = self.context_query_attn_layer(C, Q, mask_C, mask_Q)
        x = self.CQ_projection(x)
        enc_1 = self.forward_stacked_enc_blocks(x, mask_C)
        enc_2 = self.forward_stacked_enc_blocks(enc_1, mask_C)
        enc_3 = self.forward_stacked_enc_blocks(enc_2, mask_C)

        logits_start = self.p_start(torch.cat((enc_1, enc_2), dim=1), mask_C)
        logits_end   = self.p_end(torch.cat((enc_1, enc_3), dim=1), mask_C)
        
        return logits_start.view(-1, self.c_limit), logits_end.view(-1, self.c_limit)
    
    
    
class ExponentialMovingAverage():
    def __init__(self, decay_rate):
        self.decay_rate = decay_rate
        self.shadow = {}
        
    def __call__(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.forward_parameter(name, param.data)

    def register_model(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.register(name, param.data)
        
    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward_parameter(self, name, x):
        assert name in self.shadow        
        new_average = (1.0 - self.decay_rate) * x  +  self.decay_rate * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
        
        
        
# ToDO: PosEncoder, AttnFlow, checking masking precedure (*?)
