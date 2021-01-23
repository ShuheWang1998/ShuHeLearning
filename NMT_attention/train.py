import shuhe_config as config
import utils
import vocab
from nmt_model import NMT
from optparse import OptionParser
import torch
import torch.nn as nn
import math
import time
import sys
import os
from docopt import docopt
import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def evaluate_ppl(model, dev_data_src, dev_data_tar, dev_batch_size):
    flag = model.training
    model.eval()

    batch_loss = 0
    batch_size = 0
    ppl = 0
    
    with torch.no_grad():
        for src, tar, tar_len in utils.batch_iter(dev_data_src, dev_data_tar, dev_batch_size):
            loss = -model(src, tar)
            loss = loss.sum()
            batch_loss += loss
            batch_size += tar_len
        ppl = math.exp(batch_loss/batch_size)
    if (flag):
        model.train()

    return ppl

def train():
    train_data_src, train_data_tar = utils.read_corpus(config.train_path)
    dev_data_src, dev_data_tar = utils.read_corpus(config.dev_path)
    text = vocab.Text(config.src_corpus, config.tar_corpus)
    
    parser = OptionParser()
    parser.add_option("--embed_size", dest="embed_size", default=config.embed_size)
    parser.add_option("--hidden_size", dest="hidden_size", default=config.hidden_size)
    parser.add_option("--window_size_d", dest="window_size_d", default=config.window_size_d)
    parser.add_option("--encoder_layer", dest="encoder_layer", default=config.encoder_layer)
    parser.add_option("--decoder_layers", dest="decoder_layers", default=config.decoder_layers)
    parser.add_option("--dropout_rate", dest="dropout_rate", default=config.dropout_rate)
    (options, args) = parser.parse_args()
    device = torch.device("cuda:0" if config.cuda else "cpu")
    model = NMT(text, options, device)
    #model = model.cuda()
    model = model.to(device)
    #model = nn.parallel.DistributedDataParallel(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    
    train2valid_loss = train2valid_words_num = log_loss = 0
    train2valid_num = log_tar_words_num = log_num = epoch = valid_num = train_iter = patience = 0
    begin_time = time.time()
    hist_valid_ppl = []

    print("begin training!")
    while (True):
        epoch += 1
        for src_sents, tar_sents, tar_words_num_to_predict in utils.batch_iter(train_data_src, train_data_tar, config.batch_size):
            train_iter += 1
            optimizer.zero_grad()
            batch_size = len(src_sents)

            now_loss = -model(src_sents, tar_sents)
            now_loss = now_loss.sum()
            loss = now_loss / batch_size

            loss.backward()

            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)

            optimizer.step()

            log_loss += now_loss
            train2valid_loss += now_loss

            log_num += batch_size
            train2valid_num += batch_size
            
            log_tar_words_num += tar_words_num_to_predict
            train2valid_words_num += tar_words_num_to_predict

            if (train_iter % config.log_step == 0):
                print("epoch %d, iter %d, avg_loss %.2f, ppl %.2f, train2valid_num %d, time %.2f sec" % (epoch, train_iter, log_loss / log_num, math.exp(log_loss/log_tar_words_num), train2valid_num, time.time() - begin_time), file=sys.stderr)
                log_loss = log_num = log_tar_words_num = 0
            
        if (epoch % config.valid_iter == 0):
            print("before valid epoch %d, iter %d, avg_loss %.2f, ppl %.2f, train2valid_num %d, time %.2f sec" % (epoch, train_iter, train2valid_loss / train2valid_num, math.exp(train2valid_loss/train2valid_words_num), train2valid_num, time.time() - begin_time), file=sys.stderr)
            train2valid_num = train2valid_loss = train2valid_words_num = 0
            valid_num += 1

            print("now begin validation ...", file=sys.stderr)

            eav_ppl = evaluate_ppl(model, dev_data_src, dev_data_tar, config.dev_batch_size)
            print("validation iter %d, ppl %.2f" % (train_iter, eav_ppl), file=sys.stderr)
            flag = len(hist_valid_ppl) == 0 or eav_ppl < min(hist_valid_ppl)
            if (flag):
                print("current model is the best!, save to [%s]" % (config.model_save_path), file=sys.stderr)
                model.save(os.path.join(config.model_save_path, "Best_checkpoint.pth"))
                torch.save(optimizer.state_dict(), os.path.join(config.model_save_path, "Best_optimizer.optim"))
            else:
                patience += 1
                print(f"hit patience {patience}!", file=sys.stderr)
                if (patience == config.patience):
                    print("load the best and decay the lr!", file=sys.stderr)
                    new_lr = optimizer.param_groups[0]['lr'] * float(config.lr_decay)
                    params = torch.load(os.path.join(config.model_save_path, "Best_checkpoint.pth"), map_location=lambda storage, loc: storage)
                    model.load_state_dict(params['state_dict'])
                    model = model.to(device)
                    #model = model.cuda()
                    #model = nn.parallel.DistributedDataParallel(model)
                    optimizer.load_state_dict(torch.load(os.path.join(config.model_save_path, "Best_optimizer.optim")))
                    for para in optimizer.param_groups:
                        para['lr'] = new_lr
                    patience = 0
        if (epoch == config.max_epoch):
            print("reach the maximum number of epochs!", file=sys.stderr)
            return

def test():
    test_path = config.test_path
    print(f"load test sentences from {test_path}", file=sys.stderr)
    test_data_src, test_data_tar = utils.read_corpus(test_path)
    model_path = config.checkpoint
    print(f"load model from {model_path}", file=sys.stderr)
    model = NMT.load(model_path)
    if (config.cuda):
        model = model.to(torch.device("cuda:0"))
        #model = model.cuda()
        #model = nn.parallel.DistributedDataParallel(model)
    predict = beam_search(model, test_data_src, 5, config.max_tar_length)
    for i in range(len(test_data_tar)):
        for j in range(len(test_data_tar[i])):
            test_data_tar[i][j] = model.text.tar.id2word[test_data_tar[i][j]]
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            for k in range(len(predict[i][j])):
                predict[i][j][k] = model.text.tar.id2word[predict[i][j][k]]
    bleu = 0.0
    for i in tqdm(len(test_data_tar)):
        bleu += compute_bleu(predict[i], test_data_tar[i])
    print(f"BLEU is {bleu}", file=sys.stderr)
    
def beam_search(model, test_data_src, search_size, max_tra_length):
    model.eval()
    predict = []
    with torch.no_grad():
        for src in tqdm(test_data_src, desc='test', file=sys.stderr):
            predict.append(model.beam_search(src, search_size, max_tra_length))
    return predict

def compute_bleu(predict, target):
    max_ = 0.0
    smooth = SmoothingFunction()
    for sub_predict in predict:
        max_ = max(max_, corpus_bleu([[target]], [sub_predict], smoothing_function=smooth))
    return max_

def main():
    #args = docopt(__doc__)
    torch.manual_seed(config.seed)
    if (config.cuda):
        torch.cuda.manual_seed(config.seed)
    #train()
    
    #if (args['--train']):
    train()
    #elif(args['--test']):
    test()
    #else:
    #    raise RuntimeError("invalid run mode!")
    

if __name__ == '__main__':
    main()