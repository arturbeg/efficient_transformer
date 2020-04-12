from playground.Models import Transformer
from io import open
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import time
import math
import argparse
from playground.Optim import ScheduledOptim
from torch.optim import Adam, SGD
import datetime
from data_utils import get_lm_corpus
import logging

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch LM1b Transformer Language Model')

parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--gating', type=str, default='none',
                    help='gating method to use: either moe or mog or none')

parser.add_argument('--bsz', type=int, default=32,
                    help='The batch size used by the transformer')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate')

parser.add_argument('--optimizer', type=str, default='adam',
                    help='the optimizer used to train the transformer')

args = parser.parse_args()
# args = parser.parse_args(['--gating', 'moe'])

BATCH_SIZE = args.bsz
N_LAYERS = 6
EPOCHS = 40
DROPOUT = 0.15
N_HEADS = 2
D_MODEL = 512
BPTT = 32  # seems to be the sequence length
CLIP = 0.25
LR = args.lr  # initial learning rate
LOG_INTERVAL = 128  # report interval
# path to save the final model
now = datetime.datetime.now().timestamp()
now_str = str(now)
LOG = 'model_vanilla_transformer.log' if args.gating == "none" else "model_moe_transformer.log"
SAVE = 'model_vanilla_transformer.pt' if args.gating == "none" else "model_moe_transformer.pt"
SAVE = now_str + '_' + SAVE
SAVE = './model_files/' + SAVE
LOG = now_str + '_' + LOG
# print(SAVE, flush=True)
logging.basicConfig(filename='./log_files/' + LOG, level=logging.DEBUG)
logging.info(SAVE)

DATA = './data/one-billion-words'
DATASET = 'lm1b'
VOCAB = 'word'

if torch.cuda.is_available():
    if not args.cuda:
        logging.warning("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

corpus = get_lm_corpus(datadir=DATA, dataset=DATASET, vocab=VOCAB)
ntokens = len(corpus.vocab)
vocab = corpus.vocab

tr_iter = corpus.get_iterator('train', BATCH_SIZE, BPTT,
                              device=device, ext_len=0)
va_iter = corpus.get_iterator('valid', BATCH_SIZE, BPTT,
                              device=device, ext_len=0)
te_iter = corpus.get_iterator('test', BATCH_SIZE, BPTT,
                              device=device, ext_len=0)
logging.info("Gating function is: " + str(args.gating))

model = Transformer(src_vocab=ntokens, trg_vocab=ntokens, d_model=D_MODEL, N=N_LAYERS, heads=N_HEADS, dropout=DROPOUT,
                    is_lm=True, mixing=args.gating, is_cuda=args.cuda)

if args.cuda and torch.cuda.device_count()  > 1:
    logging.info("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
    model = nn.DataParallel(model)

model.to(device)

criterion = nn.NLLLoss()  # changes depending on the last layer of the transformer

if args.optimizer == "adam":
    optimization_method = Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
elif args.optimizer == "sgd_momentum":
    optimization_method = SGD(model.parameters(), lr=LR, momentum=0.9)
elif args.optimizer == "sgd":
    optimization_method = SGD(model.parameters(), lr=LR)
else:
    raise RuntimeError("Please provide a valid optimization method: adam, sgd or sgd_momentum")

logging.info("The optimizer used is: " + args.optimizer)
optimizer = ScheduledOptim(optimizer=optimization_method,
                           init_lr=LR, d_model=D_MODEL, n_warmup_steps=4000)

# Training code
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


def evaluate(data_iter):
    model.eval()
    total_loss = 0.0
    ntokens = len(corpus.vocab)
    number_of_batches = 0

    with torch.no_grad():
        for batch, (data, target, seq_len) in enumerate(data_iter):
            targets = target.contiguous().view(-1)
            trg_mask = create_mask(data).to(device)
            output, aux_loss = model(src=None, trg=data, src_mask=None, trg_mask=trg_mask, is_lm=True, train=False)
            output = output.view(-1, ntokens)
            total_loss += criterion(output, targets).item()
            number_of_batches += 1

    return total_loss / number_of_batches


def train(data_iter):
    model.train()
    total_loss = 0.
    total_aux_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.vocab)
    for batch, (data, target, seq_len) in enumerate(data_iter):
        targets = target.contiguous().view(-1)
        trg_mask = create_mask(data).to(device)
        optimizer.zero_grad()
        output, aux_loss = model(src=None, trg=data, src_mask=None, trg_mask=trg_mask, is_lm=True)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        final_loss = loss + aux_loss
        final_loss.backward()

        optimizer.step_and_update_lr()

        total_loss += loss.item()
        total_aux_loss += aux_loss.item()

        if batch == 0:
            logging.info("Running without errors")

        if batch % LOG_INTERVAL == 0 and batch > 0:
            cur_loss = total_loss / LOG_INTERVAL  # curr loss is independent of the aux loss
            curr_aux_loss = total_aux_loss / LOG_INTERVAL

            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | batch {:5d} | lr {:06.6f} | ms/batch {:5.2f} | '
                  'loss {:10.4f} | aux_loss {:10.4f} | ppl {:10.4f}'.format(
                epoch, batch, LR,
                elapsed * 1000 / LOG_INTERVAL, cur_loss, curr_aux_loss, math.exp(cur_loss)))
            total_loss = 0.
            total_aux_loss = 0.
            start_time = time.time()


best_val_loss = None

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(data_iter=tr_iter)
    val_loss = evaluate(data_iter=va_iter)
    logging.info('-' * 89)
    logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:6.4f} | '
          'valid ppl {:10.4f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    logging.info('-' * 89)
    # Save the model if validation loss is the best we have seen so far
    if not best_val_loss or val_loss < best_val_loss:
        with open(SAVE, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss

# Run on test data.
test_loss = evaluate(data_iter=te_iter)
logging.info('-' * 89)
logging.info('| End of training | test loss {:6.4f} | test ppl {:10.4f}'.format(
    test_loss, math.exp(test_loss)))
logging.info('-' * 89)