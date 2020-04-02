# This file generates new sentences sampled from the language model

import argparse
import torch
import os
import numpy as np

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')

parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')

parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')

parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')

parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')

parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')

parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')

args = parser.parse_args()

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


if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)

model.eval()
corpus = Corpus(args.data)
ntokens = len(corpus.dictionary)

input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            output = model(src=None, trg=input, src_mask=None, trg_mask=None, is_lm=True, train=False)
            word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            input = torch.cat([input, word_tensor], 0)

            word = corpus.dictionary.idx2word[word_idx]
            outf.write(word + ('\n' if i % 20 == 19 else ' '))
            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
