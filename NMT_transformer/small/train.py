import shuhe_config as config
import utils
from vocab import Text
import torch
from nmt_model import NMT
import math
from tqdm import tqdm
import sys
import os
from nltk.translate.bleu_score import corpus_bleu
from optim import Optim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def evaluate_ppl(model, dev_data_src, dev_data_tar, dev_batch_size):
    flag = model.training
    model.eval()
    sum_word = 0
    sum_loss = 0
    with torch.no_grad():
        max_iter = int(math.ceil(utils.get_num(config.dev_path)/dev_batch_size))
        with tqdm(total=max_iter, desc="validation") as pbar:
            for batch_src, batch_tar, tar_word_num in utils.batch_iter(dev_data_src, dev_data_tar, dev_batch_size):
                now_batch_size = len(batch_src)
                batch_loss = -model(batch_src, batch_tar)
                batch_loss = batch_loss.sum()
                loss = batch_loss / now_batch_size
                sum_loss += batch_loss
                sum_word += tar_word_num
                pbar.set_postfix({"avg_pool": '{%.2f}' % (loss.item()), "ppl": '{%.2f}' % (math.exp(batch_loss.item() / tar_word_num))})
                pbar.update(1)
    if (flag):
        model.train()
    return math.exp(sum_loss.item() / sum_word)

def train():
    torch.manual_seed(1)
    if (config.cuda):
        torch.cuda.manual_seed(1)
    args = dict()
    args['embed_size'] = config.embed_size
    args['d_model'] = config.d_model
    args['nhead'] = config.nhead
    args['num_encoder_layers'] = config.num_encoder_layers
    args['num_decoder_layers'] = config.num_decoder_layers
    args['dim_feedforward'] = config.dim_feedforward
    args['dropout'] = config.dropout
    train_data_src, train_data_tar = utils.read_corpus(config.train_path)
    dev_data_src, dev_data_tar = utils.read_corpus(config.dev_path)
    text = Text(config.src_corpus, config.tar_corpus)
    device = torch.device("cuda:0" if config.cuda else "cpu")
    model = NMT(text, args, device)
    model = model.to(device)
    model.train()
    optimizer = Optim(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9), config.d_model, config.warm_up_step)

    valid_num = epoch = 0
    history_valid_ppl = []
    print("begin training!", file=sys.stderr)
    while (True):
        epoch += 1
        max_iter = int(math.ceil(utils.get_num(config.train_path)/config.train_batch_size))
        with tqdm(total=max_iter, desc="train") as pbar:
            for batch_src, batch_tar, tar_word_num in utils.batch_iter(train_data_src, train_data_tar, config.train_batch_size):
                optimizer.zero_grad()
                now_batch_size = len(batch_src)
                batch_loss = -model(batch_src, batch_tar)
                batch_loss = batch_loss.sum()
                loss = batch_loss / now_batch_size
                loss.backward()
                optimizer.step_and_updata_lr()
                pbar.set_postfix({"epoch": epoch, "avg_loss": '{%.2f}' % (loss.item()), "ppl": '{%.2f}' % (math.exp(batch_loss.item()/tar_word_num))})
                pbar.update(1)
        if (epoch % config.valid_iter == 0):
            valid_num += 1
            print("now begin validation...", file=sys.stderr)
            eval_ppl = evaluate_ppl(model, dev_data_src, dev_data_tar, config.dev_batch_size)
            print(eval_ppl)
            flag = len(history_valid_ppl) == 0 or eval_ppl < min(history_valid_ppl)
            if (flag):
                print(f"current model is the best! save to [{config.model_save_path}]", file=sys.stderr)
                history_valid_ppl.append(eval_ppl)
                model.save(os.path.join(config.model_save_path, f"new_{epoch}_{eval_ppl}_checkpoint.pth"))
                torch.save(optimizer.optimizer.state_dict(), os.path.join(config.model_save_path, f"new_{epoch}_{eval_ppl}_optimizer.optim"))
        if (epoch == config.max_epoch):
            print("reach the maximum number of epochs!", file=sys.stderr)
            return

def beam_search(model, test_data_src, search_size, max_tar_length):
    model.eval()
    predict = []
    with torch.no_grad():
        for src in tqdm(test_data_src, desc='test', file=sys.stderr):
            predict.append(model.beam_search(src, search_size, max_tar_length))
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
    model_path = config.checkpoint
    model = NMT.load(model_path)
    if (config.cuda):
        model = model.to(torch.device("cuda:0"))
    predict = beam_search(model, test_data_src, 5, config.max_tar_length)
    for i in range(len(test_data_tar)):
        for j in range(len(test_data_tar[i])):
            test_data_tar[i][j] = model.text.tar.id2word[test_data_tar[i][j]]
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            for k in range(len(predict[i][j])):
                predict[i][j][k] = model.text.tar.id2word[predict[i][j][k]]
    best_predict = []
    for i in tqdm(len(test_data_tar), desc="find best predict"):
        best_predict.append(predict[i][compare_bleu(predict[i], test_data_tar[i])])
    bleu = corpus_bleu([[tar] for tar in test_data_tar], [pre for pre in best_predict])
    print(f"Corpus BLEU: {bleu * 100}", file=sys.stderr)

def main():
    train()

if __name__ == '__main__':
    main()