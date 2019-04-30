
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.conv import Initialized_Conv1d, RegularConv
from modules.helpers import mask_logits, apply_mask

import math

# bangliu

class MultiHeadAttnBL(nn.Module):
    def __init__(self, d_model, heads, droprate):
        super().__init__()
        self.d_model = d_model
        self.num_head = heads
        self.dropout = droprate
        self.mem_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in torch.split(memory, self.d_model, dim=2)]

        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask = mask)
        return self.combine_last_two_dim(x.permute(0,2,1,3)).transpose(1, 2)

    def dot_product_attention(self, q, k ,v, bias = False, mask = None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q,k.permute(0,1,3,2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x  if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret

# normal model

class MultiHeadAttn(nn.Module):
    """
    Implements Multi-Head self-attention as defined by:
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 
    Attention is all you need. CoRR, abs/1706.03762, 2017a. URL http://arxiv.org/abs/1706.03762.
    """
    def __init__(self, d_model, heads, droprate, proj_type=2):
        """
            Multi-Head-Attention
            # Arguments
                h:         (int) number of heads
                d:         (int) dimensionality of the model
                proj_type: (int) determines the projection type (0:None, 1:nn.Linear, 2:nn.Conv1d)
        """
        super().__init__()
        self.d_model = d_model
        self.h   = heads
        self.d_h = d_model // heads
        self.proj_type = proj_type
        if proj_type == 1:
            self.projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        elif proj_type == 2:
            self.projections = nn.ModuleList([RegularConv(in_channels=d_model, out_channels=d_model, kernel_size=1, bias=False) for _ in range(3)])
        
        self.dropout = nn.Dropout(droprate)
        
    def scaledDotProduct(self, Q, K, V, mask=None):
        logits = Q @ K.transpose(-2, -1)/math.sqrt(self.d_h)
        
        if mask is not None:
            shapes = [x  if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = apply_mask(logits, mask)
            
        S = F.softmax(logits, dim=-1)
        S = self.dropout(S)
        #print(Q.shape, K.transpose(-2, -1).shape, '-->', S.shape)
        return (S @ V)
    
    def forward(self, x, mask=None):
        """
            
        """
        if self.proj_type == 1:
            x = x.transpose(1, 2)
        return self.forward_(x, x, x, mask)
    
    def project(self, x, i):
        if self.proj_type == 0:
            return x
        else:
            return self.projections[i](x)
        
    def forward_(self, Q, K, V, mask=None):
        batch_size = K.shape[0]
        
        K, Q, V = (self.project(x, i) for i, x in enumerate((K, Q, V)))
        if self.proj_type == 1:
            # linear projection
            # input (B, L, D)
            # reshape to (B, L, H, DH) - splitting up D to number of heads
            # transpose to (B, H, L, DH) for matmul
            K = K.view(batch_size, -1, self.h, self.d_h).transpose(1,2)
            Q = Q.view(batch_size, -1, self.h, self.d_h).transpose(1,2)
            V = V.view(batch_size, -1, self.h, self.d_h).transpose(1,2)


            x = self.scaledDotProduct(K, Q, V, mask)#.transpose(1, 2)
            x = x.reshape(batch_size, -1, self.d_model)
            return self.project(x, -1)
        
        else:
            # CNN and None projection
            # input (B, D, L)
            # reshape to (B, H, DH, L) - splitting up D to number of heads
            # transpose to (B, H, L, DH), transpose not needed for self-attention, needs to be checkt if Q,K,V differ
            Q = Q.view(batch_size, self.h, self.d_h, -1).transpose(-2, -1)
            K = K.view(batch_size, self.h, self.d_h, -1).transpose(-2, -1)
            V = V.view(batch_size, self.h, self.d_h, -1).transpose(-2, -1)
            x = self.scaledDotProduct(K, Q, V, mask).transpose(-2, -1)
            x = x.reshape(batch_size, self.d_model, -1) # removed transpose(-2, -1) before reshape
            return x
