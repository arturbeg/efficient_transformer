import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# TODO: unit tests?

torch.manual_seed(0)

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    # np_mask = np_mask.cuda()
    return np_mask


def create_mask(trg):
    size = trg.size(1)  # get seq_len for matrix
    np_mask = nopeak_mask(size)
    # no padding masks in here
    return np_mask  # equivalent to the trg_mask

def top_level_attention(src, mask=None, dropout=None):
    d_model = src.size(2)
    scores = torch.matmul(src, src.transpose(-2, -1)) / math.sqrt(d_model)

    if mask is not None:

        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    diag_mask = torch.eye(mask.size(1)).unsqueeze(dim=0)
    scores = scores.masked_fill(diag_mask == 1, 0.0)

    return scores

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)
        num_exp = q.size(1)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, num_exp, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, num_exp, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, num_exp, -1, self.h, self.d_k)

        # transpose to get dimensions bs * num_exp * N * sl * d_model
        k = k.transpose(2, 3)
        q = q.transpose(2, 3)
        v = v.transpose(2, 3)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(2, 3).contiguous() \
            .view(bs, num_exp, -1, self.d_model)
        output = self.out(concat)

        return output


class SparseMultiHeadAttention(nn.Module):
    def __init__(self, num_lookup_subsequences, num_experts, heads, d_model, dropout=0.1, args=None):
        assert args is not None
        super().__init__()
        # num_expert is the same a the number of subsequences
        self.d_model = d_model

        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        self.num_lookup_subsequences = num_lookup_subsequences
        self.num_experts = num_experts
        self.dense_attn = MultiHeadAttention(heads=heads, d_model=d_model, dropout=dropout)

    def generate_lookup_indices(self, x):
        x = torch.sum(input=x, dim=2)

        mask = create_mask(trg=x)
        mask = mask.to(device=self.device)

        top_level_attention_scores = top_level_attention(src=x, mask=mask)

        top_attn_scores, lookup_indices = top_level_attention_scores.topk(self.num_lookup_subsequences, dim=2)

        return lookup_indices.unsqueeze(dim=2)


    def generate_lookup_subsequences(self, main_subsequences, lookup_indices):

        index = lookup_indices.expand_as(main_subsequences)

        lookup_subsequences = torch.gather(input=main_subsequences, dim=1, index=index)

        return lookup_subsequences


    def forward(self, x):

        bsz = x.size(0)

        main_subsequences = x.view(bsz, self.num_experts, -1, self.d_model)

        lookup_indices = self.generate_lookup_indices(x=main_subsequences)

        lookup_subsequences = self.generate_lookup_subsequences(main_subsequences=main_subsequences, lookup_indices=lookup_indices)

        lookup_subsequences[:, 0, :, :] = 0.0  # TODO: implement in a neater way (padding for the first token in each subsequence)

        keys_values = torch.cat([lookup_subsequences, main_subsequences], dim=2)

        # pass everything through attention layers
        output = self.dense_attn(q=main_subsequences, k=keys_values, v=keys_values, mask=None)

        output = output.view(bsz, -1, self.d_model)

        return output

# d_model = 4
# bptt = 8
# bsz = 2

# src = torch.rand(size=(2, bptt, d_model))
# # mask = create_mask(trg=src)
# sparse_attn = SparseMultiHeadAttention(num_lookup_subsequences=1, heads=4, num_experts=4, d_model=d_model, dropout=0.0)
# output = sparse_attn(x=src)