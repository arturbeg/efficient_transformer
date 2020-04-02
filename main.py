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
from torch import autograd
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
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    # data = data.view(bsz, -1).t().contiguous()
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
                    is_lm=True, mixing=args.gating, is_cuda=args.cuda).to(device)

criterion = nn.NLLLoss()  # changes depending on the last layer of the transformer
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
optimizer = ScheduledOptim(optimizer=
                           Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                           init_lr=LR, d_model=D_MODEL, n_warmup_steps=4000)

# Training code

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    # np_mask = np_mask.cuda()
    return np_mask


def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In ", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:",
                               out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

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
    # Turn on evaluation mode which disables dropout
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


def check_for_nans(model):
    number_of_nans = 0
    for name, parameter in model.named_parameters():
        if torch.isnan(parameter).any():
            print(name)
            print(parameter)
            number_of_nans += 1

    return number_of_nans > 0


def train(train_data):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    total_aux_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(1) - 1, BPTT)):
        with autograd.detect_anomaly():
            data, targets = get_batch(train_data, i)  # data is [35, 20], targets is [700]
            trg_mask = create_mask(data).to(device)
            optimizer.zero_grad()
            for submodule in model.modules():
                submodule.register_forward_hook(nan_hook)
            output, aux_loss = model(src=None, trg=data, src_mask=None, trg_mask=trg_mask, is_lm=True)
            output = output.view(-1, ntokens)
            loss = criterion(output, targets)
            final_loss = loss + aux_loss
            final_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step_and_update_lr()

            model_has_nan = check_for_nans(model)
            if model_has_nan:
                print("Nans have been identified")

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


# look over epochs

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
        # Save the model if validation loss is the best we have seen so far
        if not best_val_loss or val_loss < best_val_loss:
            with open(SAVE, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation
            # dataset
            LR /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)