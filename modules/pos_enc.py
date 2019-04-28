import torch
import torch.nn as nn

import math 

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        matrix = self.get_pos_enc_matrix_2(dim, max_len)
        self.pos_encoding = nn.Parameter(matrix, requires_grad=False)
        self.max_len = max_len
        
    def get_pos_enc_matrix(self, dim, max_len):
        pos = torch.arange(max_len, dtype=torch.float)
        base = torch.tensor(10000.0)
        div = base.expand(dim).pow(2 * torch.arange(dim, dtype=torch.float)/dim).expand(max_len, dim)
        pos_enc_matrix = pos.unsqueeze(1).expand_as(div) / div
        
        # sine and cosine added
        pos_enc_matrix[:, 0::2] = torch.sin(pos_enc_matrix[:, 0::2])
        pos_enc_matrix[:, 1::2] = torch.cos(pos_enc_matrix[:, 1::2])
        # transposing from (max_len, dim) to (dim, max_len)
        return pos_enc_matrix.t().contiguous()
    
    def get_pos_enc_matrix_2(self, channels, length, min_timescale=1.0, max_timescale=1.0e4):
        # https://github.com/BangLiu/QANet-PyTorch/blob/master/model/QANet.py#L58
        position = torch.arange(length, dtype=torch.float)
        num_timescales = channels // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * torch.exp(
                torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
        m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
        signal = m(signal) # shape: (length, channels)
        return signal.t().contiguous()
    
    def forward(self, x):
        x_dim, x_len = x.shape[-2:]
        return x + self.pos_encoding[:x_dim,:x_len]
