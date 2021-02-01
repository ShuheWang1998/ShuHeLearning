import shuhe_config as config
import torch
from nmt_model import NMT
from tqdm import tqdm
import sys
import os
from nltk.translate.bleu_score import corpus_bleu
import math
from torch.utils.data import DataLoader
from data import Data
from vocab import Vocab
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def beam_search(model, test_data, test_data_loader, search_size, max_tar_length):
    model.eval()
    predict = []
    test_tar = []
    with torch.no_grad():
        max_iter = int(math.ceil(len(test_data)/config.test_batch_size))
        with tqdm(range(max_iter), desc='test', file=sys.stderr) as pbar:
            for src, tar, _ in test_data_loader:
                now_predict = model.beam_search(src, search_size, max_tar_length, len(src))
                for sub_tar in tar:
                    test_tar.append(sub_tar)
                for sub in now_predict:
                    predict.append(sub)
                pbar.update(1)
    return predict, test_tar

def compare_bleu(predict, target):
    max_ = 0.0
    id_ = 0
    for i, sub_predict in enumerate(predict):
        bleu = corpus_bleu([[target]], [sub_predict])
        if (bleu > max_):
            max_ = bleu
            id_ = i
    return id_

def test():
    vocab = Vocab(config.corpus)
    print(f"load test sentences from [{config.test_path_src}], [{config.test_path_tar}]", file=sys.stderr)
    #test_data_src, test_data_tar = utils.read_corpus(config.test_path)
    test_data = Data(config.test_path_src, config.test_path_tar, vocab)
    test_data_loader = DataLoader(dataset=test_data, batch_size=config.test_batch_size, shuffle=True, collate_fn=utils.get_batch)
    model_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_transformer/result/01.29_15_17.08519745870326_checkpoint.pth"
    model = NMT.load(model_path)
    if (config.cuda):
        model = model.to(torch.device("cuda:0"))
    predict, test_data_tar = beam_search(model, test_data, test_data_loader, 20, config.max_tar_length)
    for i in range(len(test_data_tar)):
        for j in range(len(test_data_tar[i])):
            test_data_tar[i][j] = model.vocab.id2word[test_data_tar[i][j]]
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            predict[i][j] = model.vocab.id2word[predict[i][j]]
    bleu = corpus_bleu([[tar[1:-1]] for tar in test_data_tar], [pre for pre in predict])
    print(f"Corpus BLEU: {bleu * 100}", file=sys.stderr)

def main():
    test()

if __name__ == '__main__':
    main()