import torch
import torch.nn as nn
from playground.Sublayers import FeedForward, MultiHeadAttention, Norm
from playground.moe_attention import MoeMultiHeadAttention
from playground.sparse_attention import SparseMultiHeadAttention

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
    def __init__(self, d_model, heads, dropout=0.1, is_lm=True, mixing="none", ff_gating="none", is_cuda=True, num_experts=4, k=2, is_odd_layer=True, args=None):
        super().__init__()
        self.args = args
        self.mixing = mixing
        self.is_cuda = is_cuda
        self.device = torch.device("cuda" if is_cuda else "cpu")
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        if mixing == "none":
            if self.args.sparse_attn:
                self.attn_1 = SparseMultiHeadAttention(num_lookup_subsequences=1, num_experts=32, heads=heads, d_model=d_model, dropout=0.1, is_cuda=is_cuda)
            else:
                self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        elif mixing == "moe":
            self.attn_1 = MoeMultiHeadAttention(d_model, heads, num_experts=num_experts, k=k, dropout=dropout,
                                                is_cuda=is_cuda)
        else:
            raise Exception("Please provide a valid mixing method! Current options are none and moe")

        self.ff = FeedForward(d_model, dropout=dropout, ff_gating=ff_gating, num_experts=num_experts, k=k, is_cuda=is_cuda, is_odd_layer=is_odd_layer) # introduce moe feedforward

        if not is_lm:
            self.norm_3 = Norm(d_model)
            self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
            self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask, is_lm=True, train=True, performLogging=False):
        aux_loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True).to(self.device) # TODO: maybe get rid of this
        if is_lm and self.mixing == "none":
            x2 = self.norm_1(x)

            if self.args.sparse_attn:
                x = x + self.dropout_1(self.attn_1(x=x2))
            else:
                x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))

            x2 = self.norm_2(x)
            x2, additional_loss = self.ff(x2, performLogging=performLogging)
            aux_loss = aux_loss + additional_loss
            x = x + self.dropout_2(x2) # x + refers to a residual connectinon
        elif is_lm and self.mixing == "moe":
            x2 = self.norm_1(x)
            attn_out, additional_loss = self.attn_1(x2, x2, x2, mask=trg_mask, train=train)
            aux_loss = aux_loss + additional_loss
            x = x + self.dropout_1(attn_out)
            x2 = self.norm_2(x)
            x2, additional_loss = self.ff(x2, performLogging=performLogging)
            aux_loss = aux_loss + additional_loss
            x = x + self.dropout_2(x2)
        else:
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
                                               src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))

        return x, aux_loss