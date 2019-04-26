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




class Embedding(nn.Module):
    def __init__(self, wemb_dim, cemb_dim, d_model,
                 dropout_w=0.1, dropout_c=0.05):
        super().__init__()
        self.conv2d = nn.Conv2d(cemb_dim, d_model, kernel_size = (1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(wemb_dim + d_model, d_model, bias=False)
        self.high = Highway(2, d_model)
        self.dropout_w = dropout_w
        self.dropout_c = dropout_c

    def forward(self, ch_emb, wd_emb, length):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)

        wd_emb = F.dropout(wd_emb, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb


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
            Where H represents the context paragraph and U represents the query
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
        
    def forward(self, S, U, mask):
        """Expected shapes for input tensors:
            S:  (B, T, J)
            U:  (B, J, D)
           U is synonym to query
           Where B stands for batch, T for context length, J for query length and D for embedding dimension.
        """
        assert S.dim() == 3, f"S is required to have 3 dim (B, T, J), but got shape: {S.shape}"
        assert U.dim() == 3, f"U is required to have 3 dim (B, J, D), but got shape: {U.shape}"
        assert S.shape[2] == U.shape[1], f"Dimension mismatch for J, got (S and U) {S.shape[2]} and  {U.shape[1]}"
        S_ = F.softmax(apply_mask(S, mask), dim=2)
        return torch.einsum('btj,bjd->btd', S_, U)
    
    
class DCNAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, S, C, mask_C, mask_Q):
        S_  = F.softmax(apply_mask(S, mask_Q), dim=2)
        S__ = F.softmax(apply_mask(S, mask_C), dim=1)
        return S_ @ S__.transpose(2, 1) @ C
    
class ContextQueryAttention(nn.Module):
    def __init__(self, d_model, droprate=0.0):
        super().__init__()
        
        self.attn_flow_layer = AttentionFlowLayer(d_model=d_model)
        self.c2q_layer       = Context2Query()
        self.q2c_layer       = DCNAttention()
        self.dropout         = nn.Dropout(p=droprate)
        
    def forward(self, C, Q, mask_C, mask_Q):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)

        batch_size = C.shape[0]
        mask_C = mask_C.view(batch_size, -1, 1)  # batch_size, context_limit, 1
        mask_Q = mask_Q.view(batch_size, 1, -1)  # batch_size, 1, question_limit


        S  = self.attn_flow_layer(C, Q)
        A  = self.c2q_layer(S, Q, mask_Q)
        B  = self.q2c_layer(S, C, mask_C, mask_Q)
        return self.dropout(torch.cat((C, A, C*A, C*B), dim=2)).transpose(1, 2)



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
        if train_cemb:
            self.char_emb = nn.Embedding.from_pretrained(char_mat, freeze=False)
        else:
            self.char_emb = nn.Embedding.from_pretrained(char_mat)
        self.word_emb = nn.Embedding.from_pretrained(word_mat)
        wemb_dim = word_mat.shape[1]
        cemb_dim = char_mat.shape[1]
        self.emb = Embedding(wemb_dim, cemb_dim, d_model)
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
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cc, Cw, self.Lc), self.emb(Qc, Qw, self.Lq)
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
