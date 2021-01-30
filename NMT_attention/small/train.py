import shuhe_config as config
import utils
from vocab import Text
from nmt_model import NMT
from optparse import OptionParser
import torch
import math
import sys
import os
from tqdm import tqdm
from optim import Optim
from data import Data
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def evaluate_ppl(model, dev_data, dev_loader):
    flag = model.training
    model.eval()

    batch_loss = 0
    batch_size = 0
    ppl = 0
    
    with torch.no_grad():
        max_iter = int(math.ceil(len(dev_data)/config.dev_batch_size))
        with tqdm(total=max_iter, desc="validation") as pbar:
            for src, tar, tar_len in dev_loader:
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
    text = Text(config.src_corpus, config.tar_corpus)
    train_data = Data(config.train_path_src, config.train_path_tar)
    dev_data = Data(config.dev_path_src, config.dev_path_tar)
    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, collate_fn=utils.get_batch)
    dev_loader = DataLoader(dataset=dev_data, batch_size=config.dev_batch_size, shuffle=True, collate_fn=utils.get_batch)    
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
    model_path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_attention/small/result/01.29_140_1.0110098063079365_checkpoint.pth"
    print(f"load model from {model_path}", file=sys.stderr)
    model = NMT.load(model_path)
    #model = torch.nn.DataParallel(model)
    model = model.to(device)
    model = model.cuda()
    model.train()
    optimizer = Optim(torch.optim.Adam(model.parameters()))
    #optimizer = Optim(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9), config.hidden_size, config.warm_up_step)
    #print(optimizer.lr)
    #epoch = 140
    epoch = valid_num = 0
    hist_valid_ppl = []

    print("begin training!")
    while (True):
        epoch += 1
        max_iter = int(math.ceil(len(train_data)/config.batch_size))
        with tqdm(total=max_iter, desc="train") as pbar:
            for src_sents, tar_sents, tar_words_num_to_predict in train_loader:
                optimizer.zero_grad()
                batch_size = len(src_sents)

                now_loss = -model(src_sents, tar_sents)
                now_loss = now_loss.sum()
                loss = now_loss / batch_size
                loss.backward()

                _ = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                #optimizer.updata_lr()
                optimizer.step_and_updata_lr()

                pbar.set_postfix({"epwwoch": epoch, "avg_loss": loss.item(), "ppl": math.exp(now_loss.item()/tar_words_num_to_predict), "lr": optimizer.lr})
                #pbar.set_postfix({"epoch": epoch, "avg_loss": loss.item(), "ppl": math.exp(now_loss.item()/tar_words_num_to_predict)})
                pbar.update(1)
        #print(optimizer.lr)
        if (epoch % config.valid_iter == 0):
            #if (epoch >= 15*config.valid_iter):
            if (valid_num % 3 == 0):
                optimizer.updata_lr()
                valid_num = 0
            valid_num += 1
            print("now begin validation ...", file=sys.stderr)
            eav_ppl = evaluate_ppl(model, dev_data, dev_loader)
            print("validation ppl %.2f" % (eav_ppl), file=sys.stderr)
            flag = len(hist_valid_ppl) == 0 or eav_ppl < min(hist_valid_ppl)
            if (flag and eav_ppl < 25):
                print("current model is the best!, save to [%s]" % (config.model_save_path), file=sys.stderr)
                hist_valid_ppl.append(eav_ppl)
                model.save(os.path.join(config.model_save_path, f"01.29_{epoch}_{eav_ppl}_checkpoint.pth"))
                torch.save(optimizer.optimizer.state_dict(), os.path.join(config.model_save_path, f"01.29_{epoch}_{eav_ppl}_optimizer.optim"))
        if (epoch == config.max_epoch):
            print("reach the maximum number of epochs!", file=sys.stderr)
            return

def main():
    torch.manual_seed(config.seed)
    if (config.cuda):
        torch.cuda.manual_seed(config.seed)
    train()
    
if __name__ == '__main__':
    main()