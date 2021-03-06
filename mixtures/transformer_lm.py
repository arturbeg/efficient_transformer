# TODO: clean up, get rid of unnecessary code
import torch
import copy
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
from mixtures.moe_multiheaded_attention import MoE
from mixtures.MOG_multiheaded_attention import MoG
from torch.nn import Embedding
import torch.nn.functional as F
import math


class TransformerLM(Module):
    def __init__(self, ntoken, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", custom_decoder=None, gating='moe'):
        super(TransformerLM, self).__init__()

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, gating=gating)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        # self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.gating = gating  # can either be moe or mog

        self.embedding = Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.linear = Linear(d_model, ntoken)
        self.src_mask = None

        self._reset_parameters() # init weights...




    # TODO: handle trg all in one Transformer model
    def forward(self, trg, src, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask




        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        if src.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        # No encoder in the TransformerLM
        output, aux_loss2 = self.decoder(src, tgt_mask=self.src_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        output = self.linear(output)
        output = F.log_softmax(output, dim=-1)  # TODO: -1 dimension refers to the d_model (embedding dimension)

        return output, aux_loss2

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # alternative mask implementation
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerDecoder(Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt
        aux_loss = torch.tensor(0.0, dtype=torch.float)


        for i in range(self.num_layers):
            output, new_loss = self.layers[i](output, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

            # TODO: make sure the gradient gets computed correctly
            aux_loss += new_loss

        if self.norm:
            output = self.norm(output)

        return output, aux_loss

class TransformerDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", gating="moe"):
        super(TransformerDecoderLayer, self).__init__()

        if gating == "moe":
            self.self_attn = MoE(d_model, nhead, num_experts=4, dropout=dropout)
        elif gating == "mog":
            self.self_attn = MoG(d_model, nhead, num_experts=4, dropout=dropout)
        else:
            raise Exception("Provide a valid gating procedure: can either be moe or mog, you have provided ", gating)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, aux_loss = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0:2]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # No memory in the decoder only case
        # tgt2, aux_loss2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0:2]
        # tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, aux_loss


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)