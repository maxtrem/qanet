import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFlowLayer(nn.Module):
    """
        Attention-Flow-Layer after:
        Seo, Minjoon, et al. "Bidirectional attention flow for machine comprehension." 
        arXiv preprint arXiv:1611.01603 (2016).
    """
    def __init__(self, d_model, droprate=0.1):
        super().__init__()
        self.d_model = d_model
        weight_C = torch.empty(d_model, 1)
        weight_Q = torch.empty(d_model, 1)
        weight_CmQ = torch.empty(d_model, 1, 1)
        nn.init.xavier_uniform_(weight_C)
        nn.init.xavier_uniform_(weight_Q)
        nn.init.xavier_uniform_(weight_CmQ)
        self.weight_C   = torch.nn.Parameter(weight_C)
        self.weight_Q   = torch.nn.Parameter(weight_Q)
        self.weight_CmQ = torch.nn.Parameter(weight_CmQ.squeeze())
        self.bias       = nn.Parameter(torch.zeros(1, dtype=torch.float))
        self.dropout    = nn.Dropout(droprate)


        
    def forward(self, C, Q):
        """
            Computes an attention flow for C and Q. After: w⊤ @ [c; q; c ◦ q]
            Both parameters C and Q are expected to have 3 dimensions: (batch_dim, seq_length, embed_dim)
            param C: will be used to calculate first dim of attention matrix  (t used as index letter)
            param Q: will be used to calculate second dim of attention matrix (j used as index letter)
            Where C represents the context paragraph and Q represents the query


        """
        assert C.dim()    == Q.dim() == 3, f"Both C and Q are required to have 3 dim, but got shapes: {C.shape} and {Q.shape}"
        assert C.shape[2] == Q.shape[2] == self.d_model,   f"Embedding dim needs to be equal for C, Q and weights, got ({C.shape[2]}, {Q.shape[2]}, {self.d_model})"
        assert C.shape[0] == Q.shape[0],   f"Both C and Q are required to have same batch size, but got: {C.shape[0]} != {Q.shape[0]}"
        b, t, j, d = C.shape[0], C.shape[1], Q.shape[1], C.shape[2]
        C = self.dropout(C)
        Q = self.dropout(Q)
        p1 = (C @ self.weight_C).expand([-1, -1, j])
        p2 = (Q @ self.weight_Q).transpose(1, 2).expand([-1, t, -1])
        p3 = torch.einsum('btd,d,bjd->btj', C, self.weight_CmQ, Q)
        return p1+p2+p3 + self.bias

class ContextQueryAttention(nn.Module):
    def __init__(self, d_model, droprate=0.0):
        super().__init__()
        self.attn_flow_layer = AttentionFlowLayer(d_model, droprate)
        
    def forward(self, C, Q, mask_C, mask_Q):
        C_len, Q_len = C.shape[-1], Q.shape[-1]
        
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        
        S  = self.attn_flow_layer(C, Q)
        
        mask_C = mask_C.view(-1, C_len, 1)
        mask_Q = mask_Q.view(-1, 1, Q_len)
        
        S_  = F.softmax(mask_logits(S, mask_Q), dim=2)
        S__ = F.softmax(mask_logits(S, mask_C), dim=1)
        
        # Context2Query
        A  = S_ @ Q
        # Query2Context - DCNAttention
        B  = S_ @ S__.transpose(2, 1) @ C # torch.einsum('btj,bkj,bkd->btd', S_, S__, C)
        return torch.cat((C, A, C*A, C*B), dim=2).transpose(1, 2)

