import shuhe_config as config
import utils
from nmt_model import NMT
import torch
import sys
import os
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def beam_search(model, test_data_src, search_size, max_tra_length):
    model.eval()
    predict = []
    with torch.no_grad():
        for src in tqdm(test_data_src, desc='test', file=sys.stderr):
            predict.append(model.beam_search(src, search_size, max_tra_length))
    return predict

def compare_bleu(predict, target):
    max_ = 0.0
    id_ = 0
    for i, sub_predict in enumerate(predict):
        bleu = corpus_bleu([[target]], [sub_predict])
        if (max_ < bleu):
            max_ = bleu
            id_ = i
    return id_

def test():
    test_path = config.test_path
    print(f"load test sentences from {test_path}", file=sys.stderr)
    test_data_src, test_data_tar = utils.read_corpus(test_path)
    model_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_attention/small/880_16.40980823309212_checkpoint.pth"
    print(f"load model from {model_path}", file=sys.stderr)
    model = NMT.load(model_path)
    if (config.cuda):
        model = model.to(torch.device("cuda:0"))
        #model = model.cuda()
        #model = nn.parallel.DistributedDataParallel(model)
    predict = beam_search(model, test_data_src, 5, config.max_tar_length)
    for i in range(len(test_data_tar)):
        for j in range(len(test_data_tar[i])):
            test_data_tar[i][j] = model.text.tar.id2word[model.text.tar.word_offset+test_data_tar[i][j]]
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            for k in range(len(predict[i][j])):
                predict[i][j][k] = model.text.tar.id2word[predict[i][j][k]]
    best_predict = []
    for i in tqdm(range(len(test_data_tar)), desc="find best predict"):
        best_predict.append(predict[i][compare_bleu(predict[i], test_data_tar[i])])
    bleu = corpus_bleu([[ref[1:-1]] for ref in test_data_tar], [pre for pre in best_predict])
    print(f"BLEU is {bleu*100}", file=sys.stderr)

def main():
    torch.manual_seed(config.seed)
    if (config.cuda):
        torch.cuda.manual_seed(config.seed)
    test()

if __name__ == '__main__':
    main()