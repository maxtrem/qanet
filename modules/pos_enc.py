import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        matrix = self.get_pos_enc_matrix(dim, max_len)
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
    
    def forward(self, x):
        x_dim, x_len = x.shape[-2:]
        return x + self.pos_encoding[:x_dim,:x_len]
