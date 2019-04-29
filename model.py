import torch
import torch.nn as nn
import torch.nn.functional as F

import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def apply_mask(target, mask, eps=-1e30):
    return target * mask + (1 - mask) * (eps)


#from modules.helpers import Activation
###  CNNs

from modules.conv import DepthwiseSeparableCNN, DepthwiseSeparableConv, Initialized_Conv1d, RegularConv

from modules.ffnet import FeedForward

from modules.embedding import InputEmbedding


            
    



from modules.blocks import EncoderBlock


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

        self.context_encoder       = EncoderBlock(d_model=d_model, seq_limit=c_limit, kernel_size=7, droprate=droprate, shared_norm=False)
        self.question_encoder      = EncoderBlock(d_model=d_model, seq_limit=q_limit, kernel_size=7, droprate=droprate, 
                                                  shared_weight=self.context_encoder.main_layers)
        
        self.context_query_attn_layer = ContextQueryAttention(d_model, droprate)
        
        self.CQ_projection         = Initialized_Conv1d(d_model * 4, d_model)
        
        stacked_encoder_blocks     = [EncoderBlock(d_model=d_model, seq_limit=c_limit, kernel_size=5, num_conv_layers=2, droprate=droprate) for _ in range(7)]
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
    
    
    

        
# ToDO: PosEncoder, AttnFlow, checking masking precedure (*?)
