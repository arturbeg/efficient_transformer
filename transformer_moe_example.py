# Full transformer example
# from transformer import Transformer
# import torch
# transformer_model = Transformer(nhead=4, num_encoder_layers=2)
# src = torch.rand((10, 32, 512))
# # 10 is the number of words in the sentence, 32 is the batch size and 512 is the dimensionality of a word??
# tgt = torch.rand((20, 32, 512))
# out, loss = transformer_model(src, tgt)
#
# print(out.shape)

from transformer_lm import TransformerLM
import torch

transformer_lm_model = TransformerLM(nhead=4)

src = torch.rand((10, 32, 512))
out, loss = transformer_lm_model(src)
print(out.shape)
print(loss)

