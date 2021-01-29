import shuhe_config as config
import utils
import torch
from nmt_model import NMT
from tqdm import tqdm
import sys
import os
from nltk.translate.bleu_score import corpus_bleu
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_test_batch(test_data_src, batch_size):
    len_ = len(test_data_src)
    output = []
    for i in range(len_):
        output.append(test_data_src[i])
        if ((i+1) % batch_size == 0):
            yield output
            output = []
    if (len_ % batch_size != 0):
        yield output

def beam_search(model, test_data_src, search_size, max_tar_length):
    model.eval()
    predict = []
    with torch.no_grad():
        max_iter = int(math.ceil(len(test_data_src)/config.test_batch_size))
        with tqdm(range(max_iter), desc='test', file=sys.stderr) as pbar:
            for src in get_test_batch(test_data_src, config.test_batch_size):
                now_predict = model.beam_search(src, search_size, max_tar_length, len(src))
                for pre in now_predict:
                    predict.append(pre)
                #predict.append(now_predict)
                pbar.update(1)
    return predict

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
    print(f"load test sentences from {config.test_path}", file=sys.stderr)
    test_data_src, test_data_tar = utils.read_corpus(config.test_path)
    model_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_transformer/result/01.28_25_18.580518429044915_checkpoint.pth"
    model = NMT.load(model_path)
    if (config.cuda):
        model = model.to(torch.device("cuda:0"))
    predict = beam_search(model, test_data_src, 15, config.max_tar_length)
    for i in range(len(test_data_tar)):
        for j in range(len(test_data_tar[i])):
            test_data_tar[i][j] = model.text.tar.id2word[test_data_tar[i][j]]
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            for k in range(len(predict[i][j])):
                predict[i][j][k] = model.text.tar.id2word[predict[i][j][k]]
    best_predict = []
    for i in tqdm(range(len(test_data_tar)), desc="find best predict"):
        best_predict.append(predict[i][compare_bleu(predict[i], test_data_tar[i])])
    bleu = corpus_bleu([[tar[1:-1]] for tar in test_data_tar], [pre for pre in best_predict])
    print(f"Corpus BLEU: {bleu * 100}", file=sys.stderr)

def main():
    test()

if __name__ == '__main__':
    main()