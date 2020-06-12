import torch
import tensorflow_datasets as tfds
import numpy as np

class lm1bIterator(object):

    def __init__(self, ds, bsz, bptt, device='cpu'):
        self.data = ds
        self.device = device
        self.bsz = bsz
        self.bptt = bptt

    def get_sent_stream(self):
        for ex in self.data:
            ex_adjusted = np.insert(ex['text'], 0, 40000)  # 40k is out of range of the number of tokens, so is safe
            ex_adjusted = np.append(ex_adjusted, 40001)
            processed_example = torch.from_numpy(ex_adjusted)
            yield processed_example

    def stream_iterator(self, sent_stream):
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bsz, self.bptt)
        target = torch.LongTensor(self.bsz, self.bptt)

        while True:
            # data: [bsz x bptt]
            # target: [bsz x bptt]
            data.fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        data[i, n_filled:n_filled + n_new] = streams[i][:n_new]
                        target[i, n_filled:n_filled + n_new] = streams[i][1:n_new + 1]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt


    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch



class Corpus(object):
    def get_iterator(self, split, bsz, bptt, device='cpu'):
        if split == 'train':
            ds = tfds.load("lm1b/subwords32k", split="train", data_dir="./subwords32k")
            ds = ds.as_numpy_iterator()
            data_iter = lm1bIterator(ds=ds, bsz=bsz, bptt=bptt, device=device)
        else:
            # test set
            ds = tfds.load("lm1b/subwords32k", split="test", data_dir="./subwords32k")
            ds = ds.as_numpy_iterator()
            data_iter = lm1bIterator(ds=ds, bsz=bsz, bptt=bptt, device=device)

        return data_iter


def get_lm_corpus():
    corpus = Corpus()
    return corpus


# corpus = get_lm_corpus()
# tr_iter = corpus.get_iterator()
#
# for batch, (data, target, seq_len) in enumerate(tr_iter):
#     print("Sup")
#     break
