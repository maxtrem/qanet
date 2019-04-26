# -*- coding: utf-8 -*-
"""
Main model architecture.
reference: https://github.com/andy840314/QANet-pytorch-
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)  # !!!!!!!!!!!!!!!  do we need * mask after target?

def apply_mask(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)  # !!!!!!!!!!!!!!!  do we need * mask after target?

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


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.to(device)).transpose(1, 2)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))

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


class Highway(nn.Module):
    def __init__(self, layer_num, size):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        #x: shape [batch_size, hidden_size, length]
        dropout = 0.1
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
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
        
        self.highway    = Highway(2, d_model)
        
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

class SelfAttention(nn.Module):
    """
    Implements Multi-Head self-attention as defined by:
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 
    Attention is all you need. CoRR, abs/1706.03762, 2017a. URL http://arxiv.org/abs/1706.03762.
    """
    def __init__(self, d_model, num_heads, dropout, proj_type=2):
        """
            Multi-Head-Attention
            # Arguments
                h:         (int) number of heads
                d:         (int) dimensionality of the model
                proj_type: (int) determines the projection type (0:None, 1:nn.Linear, 2:nn.Conv1d)
        """
        super().__init__()
        self.d_model = d_model
        self.h   = num_heads
        self.d_h = d_model // num_heads
        self.proj_type = proj_type
        if proj_type == 1:
            self.projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        elif proj_type == 2:
            self.projections = nn.ModuleList([Initialized_Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1) for _ in range(4)])
        
        self.dropout = nn.Dropout(dropout)
        
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
            return self.project(x, -1)


class EncoderBlock(nn.Module):
    def __init__(self, conv_num, d_model, num_head, k, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(d_model, d_model, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(d_model, num_head, dropout=dropout)
        self.FFN_1 = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.conv_num = conv_num
        self.dropout = dropout

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        dropout = self.dropout
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1,2)).transpose(1,2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual

from modules.cqattn import ContextQueryAttention

class Pointer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = Initialized_Conv1d(d_model*2, 1)
        self.w2 = Initialized_Conv1d(d_model*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        return Y1, Y2


class QANet(nn.Module):
    def __init__(self, word_mat, char_mat,
                 c_max_len, q_max_len, d_model, train_cemb=False, pad=0,
                 dropout=0.1, num_head=1):  # !!! notice: set it to be a config parameter later.
        super().__init__()

        self.emb = InputEmbedding(word_mat, char_mat, d_model)
        self.num_head = num_head
        self.emb_enc = EncoderBlock(conv_num=4, d_model=d_model, num_head=num_head, k=7, dropout=0.1)
        self.cq_att = ContextQueryAttention(d_model, dropout)
        self.cq_resizer = Initialized_Conv1d(d_model * 4, d_model)
        self.model_enc_blks = nn.ModuleList([EncoderBlock(conv_num=2, d_model=d_model, num_head=num_head, k=5, dropout=0.1) for _ in range(7)])
        self.out = Pointer(d_model)
        self.PAD = pad
        self.Lc = c_max_len
        self.Lq = q_max_len
        self.dropout = dropout

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        maskC = (torch.ones_like(Cwid) *
                 self.PAD != Cwid).float()
        maskQ = (torch.ones_like(Qwid) *
                 self.PAD != Qwid).float()
        C, Q = self.emb(Cwid, Ccid), self.emb(Qwid, Qcid)
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Trainable parameters:', params)


if __name__ == "__main__":
    torch.manual_seed(12)
    test_EncoderBlock = False
    test_QANet = True
    test_PosEncoder = False

    if test_EncoderBlock:
        batch_size = 32
        seq_length = 20
        hidden_dim = 96
        x = torch.rand(batch_size, seq_length, hidden_dim)
        m = EncoderBlock(4, hidden_dim, 8, 7, seq_length)
        y = m(x, mask=None)

    if test_QANet:
        # device and data sizes
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        wemb_vocab_size = 5000
        wemb_dim = 300
        cemb_vocab_size = 94
        cemb_dim = 64
        d_model = 96
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
        for i in range(batch_size):
            question_wids[i, 0:question_lengths[i]] = \
                torch.LongTensor(1, question_lengths[i]).random_(
                    1, wemb_vocab_size)
            question_cids[i, 0:question_lengths[i], :] = \
                torch.LongTensor(1, question_lengths[i], char_dim).random_(
                    1, cemb_vocab_size)
            context_wids[i, 0:context_lengths[i]] = \
                torch.LongTensor(1, context_lengths[i]).random_(
                    1, wemb_vocab_size)
            context_cids[i, 0:context_lengths[i], :] = \
                torch.LongTensor(1, context_lengths[i], char_dim).random_(
                    1, cemb_vocab_size)

        # test whole QANet
        num_head = 1
        qanet = QANet(wv_tensor, cv_tensor,
                      c_max_len, q_max_len, d_model, train_cemb=False, num_head=num_head)
        p1, p2 = qanet(context_wids, context_cids,
                       question_wids, question_cids)
        print(p1.shape)
        print(p2.shape)

    if test_PosEncoder:
        m = PositionalEncoding(d_model=6, max_len=10, dropout=0)
        input = torch.randn(3, 10, 6)
        output = m(input)
        print(output)
        output2 = PosEncoder(input)
        print(output2)
