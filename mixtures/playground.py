'''
The inputs to the encoder will be the English sentence, and the
Outputs entering the decoder will be the French sentence

Five processes:
- Embedding the inputs
- The positional encodings
- Creating Masks
- The Multi-Headed Attention layer
- The Feed-Forward Layer
'''

from torch import nn
import torch

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)

# When each word is fed into the network, this code will perform a
# lookup and retreive its embedding vector; These vectors will then be
# learned as a parameters by the model, adjusted with each iteration
# of gradient descent

