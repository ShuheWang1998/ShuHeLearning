import sys



from docopt import docopt
from nmt_model import Hypothesis, NMT
import numpy as np
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, Text

import torch
import torch.nn.utils


def evaluate_ppl(model, dev_data, batch_size=128):
    was_training = model.training
    model.eval()
    cum_loss = 0.
    cum_tgt_words = 0.
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()
            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            cum_tgt_words += tgt_word_num_to_predict
        ppl = np.exp(cum_loss / cum_tgt_words)
    if was_training:
        model.train()
    return ppl

def train(args):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')
    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    train_batch_size = int(args['--batch-size'])
    model_save_path = args['--save']
    vocab = Vocab.load(args['--vocab'])
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)
    model.train()
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    


def beam_search(model, test_data_src, beam_size, max_decoding_time_step):
    was_training = model.training
    model.eval()
    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)
    if was_training: 
        model.train(was_training)
    return hypotheses


def main():
    args = docopt(__doc__)
    train(args)


if __name__ == '__main__':
    main()
