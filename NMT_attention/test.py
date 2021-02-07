import shuhe_config as config
import utils
from nmt_model import NMT
import torch
import sys
import os
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from data import Data
from torch.utils.data import DataLoader
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def beam_search(model, test_data, test_data_loader, search_size, max_tra_length):
    model.eval()
    predict = []
    test_data_tar = []
    with torch.no_grad():
        max_iter = int(math.ceil(len(test_data)/config.test_batch_size))
        with tqdm(total=max_iter, desc="test") as pbar:
            for src, tar, _ in test_data_loader:
                now_predict = model.beam_search(src, search_size, max_tra_length, config.test_batch_size)
                for sub in tar:
                    test_data_tar.append(sub)
                for sub in now_predict:
                    predict.append(sub)
                pbar.update(1)
    return predict, test_data_tar

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
    print(f"load test sentences from [{config.test_path_src}], [{config.test_path_tar}]", file=sys.stderr)
    test_data = Data(config.test_path_src, config.test_path_tar)
    test_data_loader = DataLoader(dataset=test_data, batch_size=config.test_batch_size, shuffle=True, collate_fn=utils.get_batch)
    model_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_attention/result/02.07_layer3drop0.4_6_8.272545151154294_checkpoint.pth"
    print(f"load model from {model_path}", file=sys.stderr)
    model = NMT.load(model_path)
    if (config.cuda):
        model = model.to(torch.device("cuda:0"))
        #model = model.cuda()
        #model = nn.parallel.DistributedDataParallel(model)
    predict, test_data_tar = beam_search(model, test_data, test_data_loader, 15, config.max_tar_length)
    for i in range(len(test_data_tar)):
        for j in range(len(test_data_tar[i])):
            test_data_tar[i][j] = model.text.tar.id2word[test_data_tar[i][j]]
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            predict[i][j] = model.text.tar.id2word[predict[i][j]]
    best_predict = []
    for i in tqdm(range(len(test_data_tar)), desc="find best predict"):
        best_predict.append(predict[i][compare_bleu(predict[i], test_data_tar[i])])
    bleu = corpus_bleu([[ref[1:-1]] for ref in test_data_tar], [pre for pre in predict])
    print(f"BLEU is {bleu*100}", file=sys.stderr)

def main():
    torch.manual_seed(config.seed)
    if (config.cuda):
        torch.cuda.manual_seed(config.seed)
    test()

if __name__ == '__main__':
    main()