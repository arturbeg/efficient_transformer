from playground.Models import Transformer
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
from data_utils_subword import get_lm_corpus
import logging

# TODO: Try to have MoE FFN in every other layer like in GShard
# TODO: Find out what hyperparameters, lr, etc they used for the Transformer in GShard
# TODO: later refactor (MoE interface --> abstract class to turn any layer into MoE)
# TODO: provide the number of experts and k to the MoE FFN
# TODO: implement ENUM gating (lists all possible ways to perform gating, all different components?)
# TODO: implement random gating as a baseline
# TODO: alternative learning schedule when training for moe, maybe need to wait for more epochs for it to learn
# TODO: incorporate GShard tweaks
# TODO: need many more experts (might need access to GCloud) --> GShard paper uses 512 experts with top 2 gating..

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch LM1b Transformer Language Model')

parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--debug', action='store_true',
                    help='debugging')

parser.add_argument('--gating', type=str, default='none',
                    help='gating method to use: either moe or mog or none')

parser.add_argument('--ff-gating', type=str, default='moe',
                    help='token level gating for the feed forward layer')

parser.add_argument('--decoder-mixing', type=str, default='none',
                    help='moe for the decoder layer in Transformer LM')

parser.add_argument('--num-epochs', type=int, default=10,
                    help='Number of epochs')

parser.add_argument('--bsz', type=int, default=64,
                    help='The batch size used by the transformer')

parser.add_argument('--bptt', type=int, default=128,
                    help='The sequence length')

parser.add_argument('--n-heads', type=int, default=4,
                    help="Number of heads in the multi headed attention layer")

parser.add_argument('--num-experts', type=int, default=16,
                    help='Total number of experts')

parser.add_argument('--k', type=int, default=2,
                    help='Number of experts gated through')

parser.add_argument('--d-model', type=int, default=256,
                    help='Embedding dimension')

parser.add_argument('--lr', type=float, default=1.0,
                    help='Initial learning rate')

parser.add_argument('--optimizer', type=str, default='adam',
                    help='the optimizer used to train the transformer')

args = parser.parse_args()

DEBUG = args.debug
NTOKENS = 32711 + 2  # lm1b/subwords32k (+ start and stop token)
NUM_EXPERTS = args.num_experts # total number of experts
K = args.k # experts used
BATCH_SIZE = args.bsz
N_LAYERS = 3
EPOCHS = args.num_epochs
DROPOUT = 0.1
N_HEADS = args.n_heads
D_MODEL = args.d_model
BPTT = args.bptt
LR = args.lr  # initial learning rate
WARMUP = 4000
LOG_INTERVAL = 128  # report interval
# path to save the final model
now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
now_str = str(now)
LOG = 'model_vanilla_transformer.log' if args.gating == "none" else "model_moe_transformer.log"
SAVE = 'model_vanilla_transformer.pt' if args.gating == "none" else "model_moe_transformer.pt"

if args.decoder_mixing == "moe":
    LOG = "model_moe_decoder.log"
    SAVE = "model_moe_decoder.pt"
    logging.info("Performing decoder mixing")
elif args.ff_gating == "moe":
    LOG = "model_moe_ffn.log"
    SAVE = "model_moe_ffn.pt"
    logging.info("Performing FFN token level mixing")

SAVE = now_str + '_' + SAVE
SAVE = './model_files/' + SAVE
LOG = now_str + '_' + LOG
logging.basicConfig(filename='./log_files/' + LOG, level=logging.DEBUG)
logging.info(SAVE)
logging.info("The batch size is: " + str(BATCH_SIZE))
logging.info("Number of epochs is : " + str(EPOCHS))
logging.info("The context length is : " + str(BPTT))
logging.info("D_model is : " + str(D_MODEL))
logging.info("Number of attention heads is : " + str(N_HEADS))
logging.info("Number of decoder layers is : " + str(N_LAYERS))
logging.info("Initial learning rate is : " + str(LR))
logging.info("Number of warmup steps is : " + str(WARMUP))
logging.info("k is : " + str(K))
logging.info("Number of experts is : " + str(NUM_EXPERTS))

if DEBUG:
    N_LAYERS = 2
    D_MODEL = 32
    LOG_INTERVAL = 2
    BPTT = 32
    N_HEADS = 2

if torch.cuda.is_available():
    if not args.cuda:
        logging.warning("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if args.cuda else "cpu")

corpus = get_lm_corpus(number_of_epochs=EPOCHS)
ntokens = NTOKENS

te_iter = corpus.get_iterator('test', BATCH_SIZE, BPTT,
                              device=device)

logging.info("Gating function is: " + str(args.gating))

model = Transformer(src_vocab=ntokens, trg_vocab=ntokens, d_model=D_MODEL, N=N_LAYERS, heads=N_HEADS, dropout=DROPOUT,
                    is_lm=True, mixing=args.gating, is_cuda=args.cuda, decoder_mixing=args.decoder_mixing, num_experts=NUM_EXPERTS, k=K,
                    ff_gating=args.ff_gating)

if args.cuda and torch.cuda.device_count() > 1:
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
                           init_lr=LR, d_model=D_MODEL, n_warmup_steps=WARMUP)

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
    ntokens = NTOKENS
    number_of_batches = 0

    with torch.no_grad():
        for batch, (data, target, seq_len) in enumerate(data_iter):
            targets = target.contiguous().view(-1).to(device)
            trg_mask = create_mask(data).to(device)
            output, aux_loss = model(src=None, trg=data, src_mask=None, trg_mask=trg_mask, is_lm=True, train=False)
            output = output.view(-1, ntokens)
            total_loss += criterion(output, targets).item()
            number_of_batches += 1

    return total_loss / number_of_batches

def train(epoch_counter):
    logging.info("Current Epoch is: " + str(epoch_counter))
    tr_iter = corpus.get_iterator('train', BATCH_SIZE, BPTT,
                                  device=device)
    model.train()
    total_loss = 0.
    total_aux_loss = 0.
    start_time = time.time()
    ntokens = NTOKENS
    for batch, (data, target, seq_len) in enumerate(tr_iter):
        if DEBUG:
            print(data)
        targets = target.contiguous().view(-1).to(device)
        trg_mask = create_mask(data).to(device)
        data = data.to(device)  # TODO: data_utils_subword (to device)
        optimizer.zero_grad()
        output, aux_loss = model(src=None, trg=data, src_mask=None, trg_mask=trg_mask, is_lm=True)
        if DEBUG:
            logging.info("Output dimensions: " + str(output.size()))
            logging.info("Targets dimensions: " + str(targets.size()))
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        final_loss = loss + aux_loss
        final_loss.backward()

        performLogging = (batch % LOG_INTERVAL == 0 and batch > 0)

        optimizer.step_and_update_lr(performLogging=performLogging)

        total_loss += loss.item()
        total_aux_loss += aux_loss.item()

        if batch == 0:
            logging.info("Running without errors")

        if performLogging:
            cur_loss = total_loss / LOG_INTERVAL  # curr loss is independent of the aux loss
            curr_aux_loss = total_aux_loss / LOG_INTERVAL

            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | batch {:5d} | ms/batch {:5.2f} | '
                  'loss {:10.4f} | aux_loss {:10.4f} | ppl {:10.4f}'.format(
                epoch_counter, batch, elapsed * 1000 / LOG_INTERVAL, cur_loss, curr_aux_loss, math.exp(cur_loss)))
            total_loss = 0.
            total_aux_loss = 0.
            start_time = time.time()
            if DEBUG:
                print('| epoch {:3d} | batch {:5d} | ms/batch {:5.2f} | '
                  'loss {:10.4f} | aux_loss {:10.4f} | ppl {:10.4f}'.format(
                epoch_counter, batch, elapsed * 1000 / LOG_INTERVAL, cur_loss, curr_aux_loss, math.exp(cur_loss)))
                break

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(epoch_counter=epoch)

# Run on test data.
test_loss = evaluate(data_iter=te_iter)
logging.info('-' * 89)
logging.info('| End of training | test loss {:6.4f} | test ppl {:10.4f}'.format(
    test_loss, math.exp(test_loss)))
logging.info('-' * 89)