import torch
import torch.nn as nn
from playground.Layers import EncoderLayer, DecoderLayer
from playground.Embed import Embedder, PositionalEncoder
from playground.Sublayers import Norm
from torch.nn import functional as F
from playground.moe_decoder_layer import MoeDecoderLayer


def get_clones(d_model, heads, num_experts, k, ff_gating, dropout, is_lm, mixing, is_cuda, N, is_moe_decoder, args=None):
    modules = []
    for i in range(N):
        if is_moe_decoder:
            modules.append(
                MoeDecoderLayer(d_model=d_model, heads=heads, dropout=dropout,
                                is_lm=is_lm, mixing=mixing, is_cuda=is_cuda,
                                ff_gating=ff_gating,
                                num_experts=num_experts, k=k))
        else:
            is_odd_layer = (i + 1) % 2 != 0
            modules.append(DecoderLayer(d_model=d_model, heads=heads,
                                        dropout=dropout, is_lm=is_lm,
                                        mixing=mixing, is_cuda=is_cuda,
                                        ff_gating=ff_gating, num_experts=num_experts,
                                        k=k, is_odd_layer=is_odd_layer, args=args))

    return nn.ModuleList(modules)

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, is_lm=True, mixing="none", ff_gating="none", is_cuda=True, decoder_mixing="none", num_experts=4, k=2, args=None):
        super().__init__()
        self.N = N
        self.is_cuda = is_cuda
        self.device = torch.device("cuda" if is_cuda else "cpu")
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.decoder_mixing = decoder_mixing

        # TODO: in here we can use MoEDecoder Layer

        if decoder_mixing == "none":
            self.layers = get_clones(d_model=d_model, heads=heads, num_experts=num_experts, k=k, ff_gating=ff_gating, dropout=dropout, is_lm=is_lm, mixing=mixing, is_cuda=is_cuda,
                                     N=N, is_moe_decoder=False, args=args)
        elif decoder_mixing == "moe":
            self.layers = get_clones(d_model=d_model, heads=heads, num_experts=num_experts, k=k, ff_gating=ff_gating, dropout=dropout, is_lm=is_lm, mixing=mixing, is_cuda=is_cuda,
                                     N=N, is_moe_decoder=True, args=args)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask, is_lm=True, train=True, performLogging=False):
        x = self.embed(trg)
        x = self.pe(x)

        aux_loss = torch.tensor(0.0, dtype=torch.float, requires_grad=True).to(self.device)
        for i in range(self.N):
            if is_lm:
                assert not e_outputs

            if self.decoder_mixing == "none":
                x, additional_loss = self.layers[i](x, e_outputs, src_mask, trg_mask, is_lm, train=train, performLogging=performLogging)
            elif self.decoder_mixing == "moe":
                x, additional_loss = self.layers[i](x, e_outputs, src_mask, trg_mask, is_lm, train=train, performLogging=performLogging)
            aux_loss = aux_loss + additional_loss
        return self.norm(x), aux_loss


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, is_lm=True, mixing="none", ff_gating="none", is_cuda=True,
                 is_debug=True, decoder_mixing="none", num_experts=4, k=2, args=None):
        super().__init__()
        if not is_lm:
            self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)

        self.is_debug = is_debug
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout, is_lm=is_lm, mixing=mixing, is_cuda=is_cuda, decoder_mixing=decoder_mixing, ff_gating=ff_gating, num_experts=num_experts, k=k, args=args)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask, is_lm=True, train=True, performLogging=False):
        if is_lm:
            e_outputs = None
        else:
            e_outputs = self.encoder(src, src_mask)
        d_output, aux_loss = self.decoder(trg, e_outputs, src_mask, trg_mask, is_lm, train=train, performLogging=performLogging)
        output = self.out(d_output)
        final_output = F.log_softmax(output, dim=-1)  # along the embedding (d_model) dimension

        if self.is_debug:
            nan_mask = torch.isnan(final_output)
            if nan_mask.any():
                print(nan_mask.nonzero())
                indices = nan_mask.nonzero()[:, 0].unique(sorted=True)
                print("Input:", output[indices])
                print("Output:", final_output[indices])
                raise RuntimeError("NaN encountered in log_softmaxs")

        return final_output, aux_loss