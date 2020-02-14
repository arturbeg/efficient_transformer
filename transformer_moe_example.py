from transformer import Transformer
import torch


transformer_model = Transformer(nhead=4, num_encoder_layers=2)

src = torch.rand((10, 32, 512))
tgt = torch.rand((10, 32, 512))
out, loss = transformer_model(src, tgt)

print(out.shape)

