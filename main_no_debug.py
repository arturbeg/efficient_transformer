from playground.Models import Transformer
import os
from io import open
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable  # depreciated
import time
import math
import argparse
from playground.Optim import ScheduledOptim
from torch.optim import Adam
import datetime

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')

parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--gating', type=str, default='none',
                    help='gating method to use: either moe or mog or none')

parser.add_argument('--lr', type=float, default=2.0,
                    help='Initial learning rate')

args = parser.parse_args()
# args = parser.parse_args(['--gating', 'moe'])

BATCH_SIZE = 32
N_LAYERS = 6
EPOCHS = 20
DROPOUT = 0.15
N_HEADS = 2
D_MODEL = 512
BPTT = 35  # seems to be the sequence length
CLIP = 0.25
LR = args.lr  # initial learning rate
LOG_INTERVAL = 128  # report interval
# path to save the final model
now = datetime.datetime.now().timestamp()
now_str = str(now)
SAVE = 'model_vanilla_transformer.pt' if args.gating == "none" else "model_moe_transformer.pt"
SAVE = now_str + '_' + SAVE
print(SAVE)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


corpus = Corpus('./data/wikitext-2')


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).contiguous()
    return data.to(device)


eval_batch_size = 10
train_data = batchify(corpus.train, BATCH_SIZE)
val_data = batchify(corpus.valid, BATCH_SIZE)
test_data = batchify(corpus.test, eval_batch_size)

# Build the model
ntokens = len(corpus.dictionary)

print("Gating function is: ", args.gating)
model = Transformer(src_vocab=ntokens, trg_vocab=ntokens, d_model=D_MODEL, N=N_LAYERS, heads=N_HEADS, dropout=DROPOUT,
                    is_lm=True, mixing=args.gating, is_cuda=args.cuda, is_debug=False).to(device)

criterion = nn.NLLLoss()  # changes depending on the last layer of the transformer
optimizer = ScheduledOptim(optimizer=
                           Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                           init_lr=LR, d_model=D_MODEL, n_warmup_steps=4000)


# Training code
def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    return np_mask


def create_mask(trg):
    size = trg.size(1)  # get seq_len for matrix
    np_mask = nopeak_mask(size)
    # no padding masks in here
    return np_mask  # equivalent to the trg_mask


def get_batch(source, i):
    seq_len = min(BPTT, source.size(1) - 1 - i)
    data = source[:, i:i + seq_len]
    target = source[:, i + 1:i + 1 + seq_len].contiguous().view(-1)
    return data, target


def evaluate(data_source):
    model.eval()

    total_loss = 0.0

    ntokens = len(corpus.dictionary)

    with torch.no_grad():
        """Temporarily sets all the requires_grad flag to false"""
        for i in range(0, data_source.size(1) - 1, BPTT):
            data, targets = get_batch(data_source, i)

            trg_mask = create_mask(data).to(device)  # make sure there are three dimensions

            output, aux_loss = model(src=None, trg=data, src_mask=None, trg_mask=trg_mask, is_lm=True, train=False)
            output = output.view(-1, ntokens)
            total_loss += criterion(output, targets).item()

    return total_loss / len(list(range(0, data_source.size(1) - 1, BPTT)))


def train(train_data):
    model.train()
    total_loss = 0.
    total_aux_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(1) - 1, BPTT)):
        data, targets = get_batch(train_data, i)  # data is [35, 20], targets is [700]
        trg_mask = create_mask(data).to(device)
        optimizer.zero_grad()
        output, aux_loss = model(src=None, trg=data, src_mask=None, trg_mask=trg_mask, is_lm=True)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        final_loss = loss + aux_loss
        final_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step_and_update_lr()

        total_loss += loss.item()
        total_aux_loss += aux_loss.item()

        if batch == 0:
            print("Running without errors")

        if batch % LOG_INTERVAL == 0 and batch > 0:
            cur_loss = total_loss / LOG_INTERVAL  # curr loss is independent of the aux loss
            curr_aux_loss = total_aux_loss / LOG_INTERVAL

            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | aux_loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, train_data.size(1) // BPTT, LR,
                              elapsed * 1000 / LOG_INTERVAL, cur_loss, curr_aux_loss, math.exp(cur_loss)))
            total_loss = 0.
            total_aux_loss = 0.
            start_time = time.time()


best_val_loss = None

try:
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_data=train_data)
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))

        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open(SAVE, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
