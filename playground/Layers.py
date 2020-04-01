import torch
import torch.nn as nn
from playground.Sublayers import FeedForward, MultiHeadAttention, Norm
from playground.moe_attention import MoeMultiHeadAttention

DEFAULT_NUMBER_OF_EXPERTS = 4

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, is_lm=True, mixing="none", is_cuda=True):
        super().__init__()
        self.mixing = mixing
        self.is_cuda = is_cuda
        self.device = torch.device("cuda" if is_cuda else "cpu")
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        if mixing == "none":
            self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        elif mixing == "moe":
            self.attn_1 = MoeMultiHeadAttention(d_model, heads, num_experts=DEFAULT_NUMBER_OF_EXPERTS, dropout=dropout,
                                                is_cuda=is_cuda)
        else:
            raise Exception("Please provide a valid mixing method! Current options are none and moe")

        self.ff = FeedForward(d_model, dropout=dropout)

        if not is_lm:
            self.norm_3 = Norm(d_model)
            self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask, is_lm=True, train=True):
        aux_loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True).to(self.device)
        if is_lm and self.mixing == "none":
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.ff(x2))
        elif is_lm and self.mixing == "moe":
            x2 = self.norm_1(x)
            attn_out, additional_loss = self.attn_1(x2, x2, x2, mask=trg_mask, train=train)
            aux_loss = aux_loss + additional_loss
            x = x + self.dropout_1(attn_out)
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.ff(x2))
        else:
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
                                               src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))

        return x, aux_loss
