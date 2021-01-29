import shuhe_config as config
import utils
import vocab
from nmt_model import NMT
from optparse import OptionParser
import torch
import math
import sys
import os
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from optim import Optim

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def evaluate_ppl(model, dev_data_src, dev_data_tar, dev_batch_size):
    flag = model.training
    model.eval()

    batch_loss = 0
    batch_size = 0
    ppl = 0
    
    with torch.no_grad():
        max_iter = int(math.ceil(utils.get_num(config.dev_path)/config.dev_batch_size))
        with tqdm(total=max_iter, desc="validation") as pbar:
            for src, tar, tar_len in utils.batch_iter(dev_data_src, dev_data_tar, dev_batch_size):
                loss = -model(src, tar)
                loss = loss.sum()
                batch_loss += loss
                batch_size += tar_len
                pbar.update(1)
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
    #model_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_attention/result/100_183.1992118469762_checkpoint.pth"
    #print(f"load model from {model_path}", file=sys.stderr)
    #model = NMT.load(model_path)
    #model = torch.nn.DataParallel(model)
    model = model.to(device)
    #model = model.cuda()
    model.train()
    optimizer = Optim(torch.optim.Adam(model.parameters()))
    epoch = 0
    hist_valid_ppl = []
    patience_loss = patience_num = patience = 0
    patience_list = []

    print("begin training!")
    while (True):
        epoch += 1
        max_iter = int(math.ceil(utils.get_num(config.train_path)/config.batch_size))
        with tqdm(total=max_iter, desc="train") as pbar:
            for src_sents, tar_sents, tar_words_num_to_predict in utils.batch_iter(train_data_src, train_data_tar, config.batch_size):
                optimizer.zero_grad()
                batch_size = len(src_sents)

                now_loss = -model(src_sents, tar_sents)
                now_loss = now_loss.sum()
                loss = now_loss / batch_size
                patience_loss += now_loss
                patience_num += batch_size

                loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                optimizer.step_and_updata_lr()

                pbar.set_postfix({"epoch": epoch, "avg_loss": loss.item(), "ppl": math.exp(now_loss.item()/tar_words_num_to_predict)})
                pbar.update(1)
            patience_ppl = patience_loss / patience_num
            if (len(patience_list) == 0 or patience_ppl < min(patience_list)):
                patience_list.append(patience_ppl)
                patience = patience_loss = patience_num = 0
            else:
                patience += 1
                if (patience >= config.patience):
                    optimizer.updata_lr()
                    patience = 0
                    patience_list = []
                patience_loss = patience_num = 0
        print(optimizer.lr)
        if (epoch % config.valid_iter == 0):
            print("now begin validation ...", file=sys.stderr)
            eav_ppl = evaluate_ppl(model, dev_data_src, dev_data_tar, config.dev_batch_size)
            print("validation ppl %.2f" % (eav_ppl), file=sys.stderr)
            flag = len(hist_valid_ppl) == 0 or eav_ppl < min(hist_valid_ppl)
            if (flag):
                print("current model is the best!, save to [%s]" % (config.model_save_path), file=sys.stderr)
                model.save(os.path.join(config.model_save_path, f"{epoch}_{eav_ppl}_checkpoint.pth"))
                torch.save(optimizer.optimizer.state_dict(), os.path.join(config.model_save_path, f"{epoch}_{eav_ppl}_optimizer.optim"))
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
    #test()
    #else:
    #    raise RuntimeError("invalid run mode!")
    

if __name__ == '__main__':
    main()