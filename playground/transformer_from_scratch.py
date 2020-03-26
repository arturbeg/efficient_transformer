# '''
# - embedding the imputs
# - the positional encodings
# - creating masks
# - the multi-headed attention layer
# - the feed-forward layer
# '''
# from torch import nn
# import torch
# import math
# from torch.autograd import Variable
# from torch.nn import functional as F
# import copy
# import time
# # Embedding
#
#
# '''
#     Feeding the network with far more information about words than a one hot encoding would.
# '''
#
# class Embedder(nn.Module):
#     '''
#         When each word is fed into the network, this code will perform a look-up and retreive
#         its embedding vector. These vectors will then be learnt as parameters by the model,
#         adjusted with each iteration of gradient descent.
#     '''
#     def __init__(self, vocab_size, d_model):
#         super().__init__()
#         self.embed = nn.Embedding(vocab_size, d_model)
#
#     def forward(self, x):
#         return self.embed(x)
#
#
# # Giving our words context: the positional encoding
# # creating a constant of position specific values
# '''
#     A positional encoding matrix is a constant whose values are defined by the equations.
#     When added to the embedding matrix, each word embedding is altered in a way specific to its position.
# '''
#
# class PositionalEncoder(nn.Module):
#     def __init__(self, d_model, max_seq_len = 80):
#         super().__init__()
#         self.d_model = d_model
#
#         # create a constant 'pe' matrix with values dependent on pos and i
#         pe = torch.zeros(max_seq_len, d_model)
#         for pos in range(max_seq_len):
#             for i in range(0, d_model, 2):
#                 pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/d_model)))
#                 pe[pos, i+1] = math.cos(pos/ (10000 ** ((2 * (i + 1))/d_model)))
#
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         # make embeddings relatively larger
#         x = x * math.sqrt(self.d_model)
#
#         # add constant to embedding
#         seq_len = x.size(1)
#
#         x = Variable(self.pe[:, :seq_len],requires_grad=False).cuda()
#         return x
#
# '''
#     The reason we increase the embedding values before addition is to make the positional
#     encoding relatively smaller. This means the original meaning in the embedding vector wont't
#     be lost when we add them together
# '''
#
#
# # Creating our Masks
#
# '''
#     Masking plays an important role in the transformer. It serves two purposes:
#     In the encoder and decoder: to zero attention outputs wherever there is just padding
#     in the input sentences
#
#     In the decoder: to prevent the decoder "peaking" ahead at the rest of the translated
#     sentence when predicting the next word
# '''
#
# # Multi-headed attention layer, each input is split into multile heads which allow
# # the network simulateneously attend to different subsections of each embedding
#
# '''
#     Split the embedding vector into N heads, so they will then have dimensions
#     batch_size * N * seq_len * (d_model / N)
#
#     (d_model / N) we will refer to as d_k
# '''
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, heads, d_model, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.d_k = d_model // heads
#         self.h = heads
#
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#
#         self.dropout = nn.Dropout(dropout)
#         self.out = nn.Linear(d_model, d_model)
#
#     def forward(self, q, k, v, mask=None):
#         bs = q.size(0)  # batch size, in my implementation it is along the dim=1
#
#         # perform a linear operation and split into h heads
#         k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
#         q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
#         v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
#
#         # transpose to get dimensions bs * h * sl * d_model
#
#         k = k.transpose(1, 2)
#         q = q.transpose(1, 2)
#         v = v.transpose(1, 2)
#
#         # calculate attention using the function we define next
#         scores = attention(q, k, v, self.d_k, mask, self.dropout)
#
#         # concatenate heads and put through final linear layer
#         concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
#         '''
#             Contiguous here means contiguous in memory. So the contiguous function
#             doesn't affect your tensor at all, it just makes sure that it is stored
#             in a contiguous chunk of memory
#         '''
#         output = self.out(concat)
#
#         return output
#
#
#
# # Calculating attention
#
# '''
#     Before we perform softmax, we apply our mask and hence reduce values where the input
#     is padding (or in the decoder, also where the input is ahead of the current word).
#     Apply dropout after softmax. The last step is doing a dot product between the result
#     so far and V.
# '''
#
# def attention(q, k, v, d_k, mask=None, dropout=None):
#     scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
#     if mask is not None:
#         mask = mask.unsqueeze(1)
#         scores = scores.masked_fill(mask == 0, -1e9)
#
#     scores = F.softmax(scores, dim=-1)
#     if dropout is not None:
#         scores = dropout(scores)
#     output = torch.matmul(scores, v)
#
#
#     '''
#         masked_fill fills elements of a tensor with value where mak is true. The
#         shape of mask must be broadcastable with the shape of the underlying tensor.
#     '''
#     return output
#
#
# '''
#     https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
#     Broadcasting Semantics - If a PyTorch operation supports broadcase, then its Tensor
#     arguments can be automatically expanded to be of equal sizes (without making copies of
#     the data).
#
#     - Two tensors are broadcastable if:
#         - each tensor has at least one dimension
#         - when iterating over the dimension sizes, starting at the trailing dimension,
#         the dimension sizes must either be equal, one of them is q or one of them does not
#         exist
#     - If two tensors are broadcastable, the resulting tensor size is calculated as:
#         - if the number of dimensions of x and y are not equal, preprend 1 to the
#         dimensions of the tensorwith fewer dimensions to make them equal length
#         - then for each dimension size, the resulting dimensiosn size is the max of the
#         sizes of x and y along that dimension
# '''
#
# # The feed forward network
#
# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_ff=2048, dropout=0.1):
#         super().__init__()
#         # We set d_ff as a default to 2048
#         self.linear_1 = nn.Linear(d_model, d_ff)
#         self.dropout = nn.Dropout(dropout)
#         self.linear_2 = nn.Linear(d_ff, d_model)
#
#     def forward(self, x):
#         x = self.dropout(F.relu(self.linear_1(x)))
#         x = self.linear_2(x)
#         return x
#
# # Normalisation
#
# '''
#     Prevents the range of values in the layers changing too much, meaning the model
#     trains faster and has better ability to generalise
# '''
#
#
# class Norm(nn.Module):
#     def __init__(self, d_model, eps=1e-6):
#         super().__init__()
#
#         self.size = d_model
#         # Create two learnable parameters to calibrate normalisation
#
#         self.alpha = nn.Parameter(torch.ones(self.size))
#         self.bias = nn.Parameter(torch.zeros(self.size))
#         self.eps = eps
#
#     def forward(self, x):
#         norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
#             (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
#
#         '''
#             Keepdim ==> the output tensor is of the same size as input except
#             in the dimension(s) dim where it is of size 1. Otherwise dim is squeezed,
#             resulting in the output tensor having 1 or len(dim) fewer dimension(s)
#         '''
#
#         return norm
#
#
# # build an encoder layer with one multi-head attention layer and one feed forward layer
#
# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, heads, dropout=0.1):
#         super().__init__()
#         self.norm_1 = Norm(d_model)
#         self.norm_2 = Norm(d_model)
#         self.attn = MultiHeadAttention(heads, d_model)
#         self.ff = FeedForward(d_model)
#         self.dropout_1 = nn.Dropout(dropout)
#         self.dropout_2 = nn.Dropout(dropout)
#
#     def forward(self, x, mask):
#         x2 = self.norm_1(x)
#         x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
#         x2 = self.norm_2(x)
#         x = x + self.dropout_2(self.self.ff(x2))
#         return x
#
#
# # build a decoder layer with two multi-head attention layers and one
# # feed-forward layer
#
# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, heads, dropout=0.1):
#         super().__init__()
#         self.norm_1 = Norm(d_model)
#         self.norm_2 = Norm(d_model)
#         self.norm_3 = Norm(d_model)
#
#         self.dropout_1 = nn.Dropout(dropout)
#         self.dropout_2 = nn.Dropout(dropout)
#         self.dropout_3 = nn.Dropout(dropout)
#
#         self.attn_1 = MultiHeadAttention(heads, d_model)
#         self.attn_1 = MultiHeadAttention(heads, d_model)
#         self.ff = FeedForward(d_model).cuda()
#
#     def forward(self, x, e_outputs, src_mask, trg_mask):
#         x2 = self.norm_1(x)
#         x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
#         x2 = self.norm_2(x)
#         x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
#                                            src_mask))
#         x2 = self.norm_3(x)
#         x = x + self.dropout_3(self.ff(x2))
#         return x
#
#
# # We can then build a convenient cloning function that can generate multiple layers:
#
# def get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
#
# '''
#     ModuleList: holds modules in a list, can be indexed like a regular python list,
#     but modules it contains are properly registered, and will be visible by all Module
#     methods
# '''
#
# class Encoder(nn.Module):
#     def __init__(self, vocab_size, d_model, N, heads):
#         super().__init__()
#         self.N = N
#         self.embed = Embedder(vocab_size, d_model)
#         self.pe = PositionalEncoder(d_model)
#         self.layers = get_clones(EncoderLayer(d_model, heads), N)
#         self.norm = Norm(d_model)
#
#     def forward(self, src, mask):
#         x = self.embed(src)
#         x = self.pe(x)
#         for i in range(self.N):
#             x = self.layers[i](x, mask)
#
#         return self.norm(x)
#
# class Decoder(nn.Module):
#     def __init__(self, vocab_size, d_model, N, heads):
#         super().__init__()
#         self.N = N
#         self.embed = Embedder(vocab_size, d_model)
#         self.pe = PositionalEncoder(d_model)
#         self.layers = get_clones(DecoderLayer(d_model, heads), N)
#         self.norm = Norm(d_model)
#
#     def forward(self, trg, e_outputs, src_mask, trg_mask):
#         x = self.embed(trg)
#         x = self.pe(x)
#         for i in range(self.N):
#             x = self.layers[i](x, e_outputs, src_mask, trg_mask)
#
#         return self.norm(x)
#
#
# class Transformer(nn.Module):
#     def __init_(self, src_vocab, trg_vocab, d_model, N, heads):
#         super().__init__()
#         self.encoder = Encoder(src_vocab, d_model, N, heads)
#         self.decoder = Decoder(trg_vocab, d_model, N, heads)
#         self.out = nn.Linear(d_model, trg_vocab)
#
#     def forward(self, src, trg, src_mask, trg_mask):
#         e_outputs = self.encoder(src, src_mask)
#         d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
#         output = self.out(d_output)
#         # don't perform softmax on the output as this will be handled automatically
#         # by our loss function
#         return output
#
#
# # Training the model
#
# '''
#     # EuroParl dataset
# '''
#
# d_model = 512
# heads = 8
# N = 6
# src_vocab = 1000
# trg_vocab = 1000
#
# model = Transformer(src_vocab, trg_vocab, d_model, N, heads)
#
# for p in model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)
#
#
# '''
#     Initialise the parameters with a range of values that stops the signal fading or
#     getting too big.
# '''
#
# optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#
# def train_model(epochs, print_every=100):
#     model.train()
#     start = time.time()
#     temp = start
#
#     total_loss = 0
#
#     for epoch in range(epochs):
#         for i, batch in enumerate(train_iter):
#             src = batch.English.transpose(0, 1)
#             trg = batch.French.tranpose(0, 1)
#
#             # the French sentence we input has all words except
#             # the last, as it is using each word to predict the next
#
#             trg_input = trg[:, :-1]
#
#             # the words we are trying to predict
#
#             targets = trg[:, 1:].contiguous().view(-1)
#
#             # create a function to make masks using mask code about
#
#             src_mask, trg_mask = create_masks(src, trg_input)
#             preds = model(src, trg_input, src_mask, trg_mask)]
#
#             optim.zero_grad()
#
#             loss = F.cross_entropy(preds.view(-1, preds.size(-1)), results, ignore_index=target_pad)
#
#             loss.backward()
#             optim.step()
#
#             total_loss += loss.data[0]
#
#
#
#
#
# # Tensting the model
#
# '''
#     Use the below function to translate sentences. We can feed sentences directly from
#     our batches, or input custom strings
#
#     Translator works by running a loop. We start off by encoding the English sentence.
#     We then feed the decoder the <sos> token index and the encoder outputs. The decoder
#     makes a prediction from the first word, and we add this to our decoder input
#     with the sos token. We rerun the loop, getting the next prediction and adding this to the
#     decoder input, until we reach the <eos> token letting us know it has finished translating.
# '''
#
#
# def translate(model, src, max_len=80, custom_string=False):
#
#     pass
#     # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
#
#
#
#
#
#
#
#
#
#
#
#
