
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.conv import Initialized_Conv1d, RegularConv

from modules.helpers import Activation, device



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
                 droprate=0.1, na_possible=False):

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
        self.na_possible = na_possible
        if na_possible:
            self.char_NA    = nn.Parameter(torch.rand(1, 1, 1, self.char_D)) # check also (1, 1, 16, self.char_D)
            self.word_NA    = nn.Parameter(torch.rand(1, 1, self.word_D))

        self.d_model      = d_model

        self.char_cnn   = RegularConv(self.char_D , d_model, kernel_size=kernel_size, 
                                      dim=char_cnn_type, activation=nn.ReLU(), bias=True)
        self.proj_cnn   = RegularConv(self.word_D + d_model, d_model, 
                                      kernel_size=1, dim=1)
        self.word_drop  = nn.Dropout(p=droprate)
        self.char_drop  = nn.Dropout(p=droprate*0.5)
        
        self.highway    = Highway(d_model, 2, droprate=droprate)
        
    def forward_chars(self, chars):

        x = self.char_embed(chars)
        N, TOK_LIM, CHAR_LIM, EMBED_DIM = x.shape

        if self.na_possible:
            x = torch.cat((x, self.char_NA.expand(N, 1, CHAR_LIM, -1)), dim=1)
        # using 1D CNN
        if self.char_cnn.dim == 1:
            # CNN Input: (N, C, L) - merge Batch and Sequence / Time dimension, transpose input (N, L, C) to (N, C, L)
            x = self.char_cnn(x.view(N*TOK_LIM, CHAR_LIM, EMBED_DIM).transpose(1, 2))
            # collapse last dimension using max
            x = x.max(dim=-1)[0]
            # transpose to match word embedding shape (switch L and D/C), 
            return x.view(N, TOK_LIM, self.d_model).transpose(1, 2)
        # using 2D CNN
        elif self.char_cnn.dim == 2:
            # permute from (N, TOK_LIM, CHAR_LIM, C) to (N, C, TOK_LIM, CHAR_LIM)
            # apply CNN, collapse last dimension using max
            return self.char_cnn(x.permute(0, 3, 1, 2)).max(dim=-1)[0]

    def forward_words(self, x):
        x = self.word_embed(x)
        if self.na_possible:
            x = torch.cat((x, self.word_NA.expand(x.shape[0], -1, -1)), dim=1)
        return x

            
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
        embedded_words = self.word_drop(self.forward_words(token_ids).transpose(1, 2))
        # concat char and word embeddings and forward projection cnn
        x = self.proj_cnn(torch.cat((embedded_words, embedded_chars), dim=1))
        return self.highway(x)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wemb_vocab_size = 5000
    wemb_dim = 300
    cemb_vocab_size = 94
    cemb_dim = 64
    d_model = 32
    batch_size = 32
    q_max_len = 50
    c_max_len = 400
    char_dim = 16

    # fake embedding
    wv_tensor = torch.rand(wemb_vocab_size, wemb_dim)
    cv_tensor = torch.rand(cemb_vocab_size, cemb_dim)

    # fake input
    question_lengths = torch.LongTensor(batch_size).random_(1, q_max_len)
    question_wids = torch.zeros(batch_size, q_max_len).long()
    question_cids = torch.zeros(batch_size, q_max_len, char_dim).long()
    context_lengths = torch.LongTensor(batch_size).random_(1, c_max_len)
    context_wids = torch.zeros(batch_size, c_max_len).long()
    context_cids = torch.zeros(batch_size, c_max_len, char_dim).long()

    embedding = InputEmbedding(wv_tensor, cv_tensor, d_model=d_model, na_possible=False)
    out = embedding(question_wids, question_cids)
    print('Output:', out.shape)
