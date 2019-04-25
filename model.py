import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def apply_mask(target, mask, eps=-1e30):
    return target * mask + (1 - mask) * (eps)

class PositionalEncoding(nn.Module):
    """
    Creates a positional encoding layer as defined by:
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.     Attention is all you need. CoRR, abs/1706.03762, 2017a. URL http://arxiv.org/abs/1706.03762.
    """
    def __init__(self, d_model, seq_limit, droprate=0.0):
        """
        # Arguments
            d_model:   (int) dimensionality of the model
            seq_limit: (int) length of the padded / truncated sequences
            droprate:  (float) sets dropout rate for dropout
        """
        super().__init__()
        with torch.no_grad():
            base = torch.tensor(10000.0)
            pe_matrix = torch.zeros(seq_limit, d_model)
            i  = torch.arange(0.0, d_model, 2, dtype=torch.float)
            pos = torch.arange(0, seq_limit, dtype=torch.float).unsqueeze(1)
            pe_matrix[:,0::2] = torch.sin(pos / base.pow(2*i / d_model))
            pe_matrix[:,1::2] = torch.cos(pos / base.pow(2*i / d_model))
            pe_matrix = pe_matrix.transpose(0, 1).unsqueeze(0).contiguous()
                    
        self.pe_matrix = nn.Parameter(pe_matrix, requires_grad=False)
        self.dropout   = nn.Dropout(p=droprate)


        
    def forward(self, x):
        return self.dropout(x + self.pe_matrix.expand_as(x))


### DWS CNN

class DepthwiseSeparableCNN(nn.Module):
    """
    Implements a depthwise separable convolutional layer as defined by:
    Lukasz Kaiser, Aidan N Gomez, and Francois Chollet. 
    Depthwise separable convolutions for neural machine translation. arXiv preprint arXiv:1706.03059, 2017.
    """
    def __init__(self, chin, chout, kernel_size=7, dim=1, activation=None, bias=True):
        """
        # Arguments
            chin:        (int) number of input channels
            chout:       (int)  number of output channels
            kernel_size: (int or tuple) size of the convolving kernel
            dim:         (int:[1, 2, 3]) type of convolution i.e. dim=2 results in using nn.Conv2d
            bias:        (bool) controlls usage of bias for convolutional layers
        """
        super().__init__()
        CNN =  getattr(nn, f'Conv{dim}d')
        self.dim = dim
        self.depthwise_cnn = CNN(chin, chin, kernel_size=kernel_size, padding=kernel_size // 2, groups=chin, bias=bias)
        self.pointwise_cnn = CNN(chin, chout, kernel_size=1, bias=bias)
        self.activation_   = activation() if activation else None

    def activation(self, x):
        return self.activation_(x) if self.activation_ else x

    def forward(self, x):
        x = self.depthwise_cnn(x)
        x = self.pointwise_cnn(x)
        x = self.activation(x)
        return x
        
class RegularConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, dim=1, activation=None, bias=False):
        super().__init__()

        self.dim = dim
        CNN =  getattr(nn, f'Conv{dim}d')
        self.conv = CNN(
            in_channels, out_channels,
            kernel_size, padding=kernel_size // 2, 
            bias=bias)
        self.activation_ = activation() if activation else None

        if self.activation_:
            nonlinearity = self.activation_.__class__.__name__.lower()
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity=nonlinearity)
        else:
            nn.init.xavier_uniform_(self.conv.weight)

    def activation(self, x):
        return self.activation_(x) if self.activation_ else x

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    
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
            self.ff_layer = DepthwiseSeparableCNN(in_features, out_features, kernel_size=5, bias=bias)
            
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

class Highway(nn.Module):
    """
    Implements a Highway network in layer form as defined by:
    Rupesh Kumar Srivastava, Klaus Greff, and Ju ̈rgen Schmidhuber. 
    Highway networks. CoRR, abs/1505.00387, 2015. URL http://arxiv.org/abs/1505.00387.
    """
    def __init__(self, d_model, num_layers=2, activation=nn.ReLU, droprate=0.0, use_cnn=True):
        
        """
        # Arguments
            d_model:      (int)   dimensionality of the model
            num_layers:   (int)   sets the number of stacked layers used for the network
            droprate:     (float) sets dropout rate for dropout
            activation:   sets the activation function for the network, valid options can be seen in
                          the "torch.nn.modules.activation" module. Use None for no activation
            use_cnn:      (bool) if True a convolutional layer is used for H and T instead of a linear layer 

        """
        super().__init__()
        self.num_layers = num_layers
        self.T = nn.ModuleList(FeedForward(d_model, d_model, use_cnn=use_cnn) for _ in range(num_layers))
        self.H = nn.ModuleList(FeedForward(d_model, d_model, use_cnn=use_cnn) for _ in range(num_layers))
        self.activation_ = activation() if activation else None
        self.dropout     = nn.Dropout(p=droprate)
        
    def activation(self, x):
        return self.activation_(x) if self.activation_ else x
        
    def forward(self, x):
        for i in range(self.num_layers):
            T = torch.sigmoid(self.T[i](x))
            H = self.activation(self.H[i](x))
            H = self.dropout(H)
            x = H * T  +  x * (1.0 - T)
        return x


class InputEmbedding(nn.Module):
    """
        InputEmbedding converts both token and character IDs into a single embedding.
        First IDs are feed into an embedding layer. Embedded characters are then projected to `d_model` dim using a CNN,
        are then concatenated with the token embeddings and projected again to `d_model` dim using another CNN.
        Finally these embeddings are feed into a two layer Highway network.
    """
    def __init__(self, word_emb_matrix, char_emb_matrix, d_model=128, kernel_size=5, freeze_word_emb=True, freeze_ch_emb=False, char_cnn_type=1, activation=nn.ReLU, word_droprate=0.1, char_droprate=0.05):
        """
        # Arguments
            char_cnn_type:   (numpy.ndarray or torch.tensor) weight matrix containing the character embeddings
            word_emb_matrix: (numpy.ndarray or torch.tensor) weight matrix containing the word embeddings
            d_model:         (int) dimensionality of the model
            kernel_size:     (int or tuple) size of the convolving kernel
            freeze_word_emb: (bool) if set True word embeddings are froozen and not optimized
            freeze_ch_emb:   (bool) if set True character embeddings are froozen and not optimized
            char_cnn_type:   (int) sets dimensionality of convolution operation
            activation:      sets the activation function for the network, valid options can be seen in
                             the "torch.nn.modules.activation" module. Use None for no activation
        """
        super().__init__()
        self.char_embed = nn.Embedding.from_pretrained(torch.tensor(char_emb_matrix), freeze=freeze_ch_emb)
        self.word_embed = nn.Embedding.from_pretrained(torch.tensor(word_emb_matrix), freeze=freeze_word_emb)
        self.char_D     = self.char_embed.embedding_dim
        self.word_D     = self.word_embed.embedding_dim
        self.d_model      = d_model
        self.activation_= activation() if activation else activation
        self.char_cnn   = DepthwiseSeparableCNN(self.char_D , d_model, kernel_size, dim=char_cnn_type)
        self.proj_cnn   = DepthwiseSeparableCNN(self.word_D + d_model , d_model, kernel_size, dim=1)
        self.word_drop  = nn.Dropout(p=word_droprate)
        self.char_drop  = nn.Dropout(p=char_droprate)
        
        self.highway    = Highway(d_model)
        
    def activation(self, x):
        return self.activation_(x) if self.activation_ else x
        
    def forward_chars(self, chars):
        # using 1D CNN
        if self.char_cnn.dim == 1:
            x = self.char_embed(chars)
            N, TOK_LIM, CHAR_LIM, EMBED_DIM = x.shape
            # CNN Input: (N, C, L) - merge Batch and Sequence / Time dimension, transpose input (N, L, C) to (N, C, L)
            x = self.char_cnn(x.view(N*TOK_LIM, CHAR_LIM, EMBED_DIM).transpose(1, 2))
            # collapse last dimension using max
            x = x.max(dim=-1)[0]
            # transpose to match word embedding shape (switch L and D/C), 
            x = self.activation(x).view(N, TOK_LIM, self.d_model).transpose(1, 2)
            return x
        # using 2D CNN
        elif self.char_cnn.dim == 2:
            x = self.char_embed(chars)
            # permute from (N, TOK_LIM, CHAR_LIM, C) to (N, C, TOK_LIM, CHAR_LIM)
            # apply CNN, collapse last dimension using max
            x = self.char_cnn(x.permute(0, 3, 1, 2)).max(dim=-1)[0]
            return self.activation(x)

            
    def forward(self, token_ids, char_ids):
        """
        # Arguments
            token_ids: (torch.LongTensor) a tensor containing token IDs
            char_ids:  (torch.LongTensor) a tensor containing token IDs
        # Result
            Returns an embedding which combines both token and character embedding, using CNN and a Highway network.
        """
        embedded_chars = self.char_drop(self.forward_chars(char_ids))
        # forward word embedding, transpose L and D 
        embedded_words = self.word_drop(self.word_embed(token_ids).transpose(1, 2))
        # concat char and word embeddings and forward projection cnn
        x = self.proj_cnn(torch.cat((embedded_words, embedded_chars), dim=1))
        x = self.activation(x)
        return self.highway(x)



class MultiHeadAttn(nn.Module):
    """
    Implements Multi-Head self-attention as defined by:
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 
    Attention is all you need. CoRR, abs/1706.03762, 2017a. URL http://arxiv.org/abs/1706.03762.
    """
    def __init__(self, d_model=128, heads=8, proj_type=1):
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
            self.projections = nn.ModuleList([DepthwiseSeparableCNN(d_model, d_model, 5) for _ in range(4)])
        
    def scaledDotProduct(self, Q, K, V):
        S = F.softmax(Q @ K.transpose(-2, -1)/math.sqrt(self.d_h), dim=-1)
        #print(Q.shape, K.transpose(-2, -1).shape, '-->', S.shape)
        return (S @ V)
    
    def forward(self, x, mask=None):
        """
            
        """
        if self.proj_type == 1:
            x = x.transpose(1, 2)
        return self.forward_(x, x, x)
    
    def project(self, x, i):
        if self.proj_type == 0:
            return x
        else:
            return self.projections[i](x)
        
    def forward_(self, Q, K, V):
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


            x = self.scaledDotProduct(K, Q, V)#.transpose(1, 2)
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
            x = self.scaledDotProduct(K, Q, V).transpose(-2, -1)
            x = x.reshape(batch_size, self.d_model, -1) # removed transpose(-2, -1) before reshape
            return self.project(x, -1)

class MultiHeadAttnC(nn.Module):
    def __init__(self, d_model=128, heads=8, droprate=0.0, proj_type=1):
        super().__init__()
        self.d_model = d_model
        self.num_head = heads
        self.dropout     = nn.Dropout(p=droprate)

        self.mem_conv = RegularConv(in_channels=d_model, out_channels=d_model*2, kernel_size=1)
        self.query_conv = RegularConv(in_channels=d_model, out_channels=d_model, kernel_size=1)

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
            logits = apply_mask(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = self.dropout(weights)
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
            
    
class ResidualBlock(nn.Module):
    """
    ResidualBlock implements the function in form of `f(layernorm(x)) + x`. 
    Dropout and activation function can be set as well.
    """
    def __init__(self, layer, shape, norm=None, droprate=0.0, activation=nn.ReLU, shared_weight=False):
        """
        # Arguments
            layer:      (instance of nn.Module with forward() implemented), layer represents function f
            shape:      (tuple of int) shape of the last two dimensions of input
            activation: sets the activation function for the network, valid options can be seen in
                        the "torch.nn.modules.activation" module. Use None for no activation
            droprate:   (float) sets dropout rate for dropout
        """

        super().__init__()
        

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
    def __init__(self, d_model=128, seq_limit=25, kernel_size=7, num_conv_layers=4, droprate=0.0, shared_weight=False):
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
        self.positional_encoding_layer = PositionalEncoding(d_model, seq_limit, droprate=0.0)

        shared_keys = {'conv_layers', 'mh_attn', 'ffnet'}

        if shared_weight:
            missing = set.difference(shared_keys, set(shared_weight.keys()))
            assert missing == set(), f'Missing modules {missing}'
            self.main_layers = shared_weight
            shared = True


        else:
            conv_layers = [DepthwiseSeparableCNN(d_model, d_model, kernel_size=kernel_size, activation=nn.ReLU) for _ in range(num_conv_layers)]
            mh_attn = MultiHeadAttnC(d_model=d_model, heads=8, proj_type=2)
            ffnet = FeedForward(d_model, d_model, activation=None)

            self.main_layers = {'conv_layers': conv_layers,
                                'mh_attn': mh_attn,
                                'ffnet': ffnet}
            shared = False

        stacked_CNN = [ResidualBlock(conv_layer, shape=shape, shared_weight=shared) for conv_layer in self.main_layers['conv_layers']]
        self.conv_blocks = nn.Sequential(*stacked_CNN)
        
        self.self_attn_block = ResidualBlock(self.main_layers['mh_attn'], shape=shape, activation=nn.ReLU, droprate=droprate, shared_weight=shared)
        
        self.feed_forward = ResidualBlock(self.main_layers['ffnet'], shape=shape, activation=None, droprate=droprate, shared_weight=shared)

        
        
    def forward(self, x, mask=None):
        x = self.positional_encoding_layer(x)
        x = self.conv_blocks(x)
        x = self.self_attn_block(x, mask=mask)
        x = self.feed_forward(x)
        return x


import torch
import torch.nn as nn
class AttentionFlowLayer(nn.Module):
    """
        Attention-Flow-Layer after:
        Seo, Minjoon, et al. "Bidirectional attention flow for machine comprehension." 
        arXiv preprint arXiv:1611.01603 (2016).
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.weight = torch.nn.Parameter(torch.rand(d_model*3))
        
    def forward(self, H, U):
        """
            Computes an attention flow for H and U. After: w⊤ @ [h; u; h ◦ u]
            Both parameters H and U are expected to have 3 dimensions: (batch_dim, seq_length, embed_dim)
            param H: will be used to calculate first dim of attention matrix  (t used as index letter)
            param U: will be used to calculate second dim of attention matrix (j used as index letter)
        """
        assert H.dim()    == U.dim() == 3, f"Both H and U are required to have 3 dim, but got shapes: {H.shape} and {U.shape}"
        assert H.shape[2] == U.shape[2] == self.d_model,   f"Embedding dim needs to be equal for H, U and W, got {H.shape[2]}, {U.shape[2]}, {self.d_model})"
        assert H.shape[0] == U.shape[0],   f"Both H and U are required to have same batch size, but got: {H.shape[0]} != {U.shape[0]}"
        b, t, j, d = H.shape[0], H.shape[1], U.shape[1], H.shape[2]
        
        # adding columns for H
        H_expand = H.unsqueeze(2).expand(b, t, j, d)
        # adding rows for U
        U_expand = U.unsqueeze(1).expand(b, t, j, d)
        # elementwise multiplication of vectors for H and U using outer product
        H_m_U = torch.einsum('btd,bjd->btjd', H, U)
        concat_out = torch.cat((H_expand, U_expand, H_m_U), dim=3)
        return torch.einsum('d,btjd->btj', self.weight, concat_out)
    
class Context2Query(nn.Module):
    """ Implements context-to-query attention after:
        Seo, Minjoon, et al. "Bidirectional attention flow for machine comprehension." 
        arXiv preprint arXiv:1611.01603 (2016).
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, S, U):
        """Expected shapes for input tensors:
            S:  (B, T, J)
            U:  (B, J, D)

           Where B stands for batch, T for context length, J for query length and D for embedding dimension.
        """
        assert S.dim() == 3, f"S is required to have 3 dim (B, T, J), but got shape: {S.shape}"
        assert U.dim() == 3, f"U is required to have 3 dim (B, J, D), but got shape: {U.shape}"
        assert S.shape[2] == U.shape[1], f"Dimension mismatch for J, got (S and U) {S.shape[2]} and  {U.shape[1]}"
        
        S_ = F.softmax(S, dim=2)
        return torch.einsum('btj,bjd->btd', S_, U)
    
    
class DCNAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, S, C):
        S_  = F.softmax(S, dim=2)
        S__ = F.softmax(S, dim=1)
        return S_ @ S__.transpose(2, 1) @ C
    
class ContextQueryAttention(nn.Module):
    def __init__(self, d_model, droprate=0.0):
        super().__init__()
        
        self.attn_flow_layer = AttentionFlowLayer(d_model=d_model)
        self.c2q_layer       = Context2Query()
        self.q2c_layer       = DCNAttention()
        self.dropout         = nn.Dropout(p=droprate)
        
    def forward(self, C, Q):
        S  = self.attn_flow_layer(C, Q)
        A  = self.c2q_layer(S, Q)
        B  = self.q2c_layer(S, C)
        return self.dropout(torch.cat((C, A, C*A, C*B), dim=2))

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
        
        self.input_embedding_layer = InputEmbedding(word_emb_matrix, char_emb_matrix, d_model=d_model, char_cnn_type=1)
        
        # ToDo: EncoderBlock shared weights compability for loading weights
        self.context_encoder       = EncoderBlock(d_model=d_model, seq_limit=c_limit, droprate=droprate)
        self.question_encoder      = EncoderBlock(d_model=d_model, seq_limit=q_limit, droprate=droprate, 
                                                  shared=self.context_encoder)
        
        self.context_query_attn_layer = ContextQueryAttention(d_model)
        
        self.CQ_projection         = DepthwiseSeparableCNN(4*d_model, d_model, kernel_size=5)
        
        stacked_encoder_blocks     = [EncoderBlock(d_model=d_model, seq_limit=c_limit, num_conv_layers=2, droprate=droprate) for _ in range(7)]
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

        mask_C = (cwids != 0).float()
        mask_Q = (qwids != 0).float()

        C = self.input_embedding_layer(cwids, ccids)
        Q = self.input_embedding_layer(qwids, qcids)
        
        C = self.context_encoder(C, mask_C).transpose(1, 2)
        Q = self.question_encoder(Q, mask_Q).transpose(1, 2)
        
        x = self.context_query_attn_layer(C, Q).transpose(1, 2)
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
        
        
        

