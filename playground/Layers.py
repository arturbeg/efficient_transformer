import torch
import torch.nn as nn
from playground.Sublayers import FeedForward, MultiHeadAttention, Norm
from mixtures.moe_multiheaded_attention import MoE

# MoE stuff (refcator)
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


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, is_lm=True, mixing="none"):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        if mixing == "none":
            self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        elif mixing == "moe":
            self.attn_1 = MoE(d_model, heads, num_experts=DEFAULT_NUMBER_OF_EXPERTS, dropout=dropout)
        else:
            raise Exception("Please provide a valid mixing method! Available ones are none or moe")

        self.ff = FeedForward(d_model, dropout=dropout)

        if not is_lm:
            self.norm_3 = Norm(d_model)
            self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask, is_lm=True):
        if is_lm:
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
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

        return x
