
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.conv import Initialized_Conv1d, RegularConv

from modules.helpers import Activation



class Highway(nn.Module):
    """
    Implements a Highway network in layer form as defined by:
    Rupesh Kumar Srivastava, Klaus Greff, and Ju ̈rgen Schmidhuber. 
    Highway networks. CoRR, abs/1505.00387, 2015. URL http://arxiv.org/abs/1505.00387.
    """
    def __init__(self, d_model, num_layers=2, activation=None, droprate=0.1, use_cnn=True):
        
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
        self.T = nn.ModuleList(RegularConv(d_model, d_model, bias=True) for _ in range(num_layers))
        self.H = nn.ModuleList(RegularConv(d_model, d_model, bias=True) for _ in range(num_layers))
        self.activation  = Activation(activation)
        self.dropout     = nn.Dropout(p=droprate)
        
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
    def __init__(self, word_emb_matrix, char_emb_matrix, d_model=128, 
                 kernel_size=(1,5), freeze_word_emb=True, freeze_ch_emb=False, char_cnn_type=2, 
                 droprate=0.1):

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

        self.char_cnn   = RegularConv(self.char_D , d_model, kernel_size=kernel_size, 
                                      dim=char_cnn_type, activation=nn.ReLU(), bias=True)
        self.proj_cnn   = RegularConv(self.word_D + d_model, d_model, 
                                      kernel_size=1, dim=1)
        self.word_drop  = nn.Dropout(p=droprate)
        self.char_drop  = nn.Dropout(p=droprate*0.5)
        
        self.highway    = Highway(d_model, 2, droprate=droprate)
        
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
            return x.view(N, TOK_LIM, self.d_model).transpose(1, 2)
        # using 2D CNN
        elif self.char_cnn.dim == 2:
            x = self.char_embed(chars)
            # permute from (N, TOK_LIM, CHAR_LIM, C) to (N, C, TOK_LIM, CHAR_LIM)
            # apply CNN, collapse last dimension using max
            return self.char_cnn(x.permute(0, 3, 1, 2)).max(dim=-1)[0]

            
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
        return self.highway(x)
