import shuhe_config as config
import torch
from nmt_model import NMT
import math
from tqdm import tqdm
import sys
import os
from optim import Optim
from data import Data
from torch.utils.data import DataLoader
from vocab import Vocab
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def evaluate_ppl(model, dev_data, dev_loader, dev_batch_size, vocab):
    flag = model.training
    model.eval()
    sum_word = 0
    sum_loss = 0
    with torch.no_grad():
        max_iter = int(math.ceil(len(dev_data)/dev_batch_size))
        with tqdm(total=max_iter, desc="validation") as pbar:
            for batch_src, batch_tar, tar_word_num in dev_loader:
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
    vocab = Vocab(config.corpus)
    train_data = Data(config.train_path_src, config.train_path_tar, vocab)
    dev_data = Data(config.dev_path_src, config.dev_path_tar, vocab)
    train_loader = DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=True, collate_fn=utils.get_batch)
    dev_loader = DataLoader(dataset=dev_data, batch_size=config.dev_batch_size, shuffle=True, collate_fn=utils.get_batch)
    #train_data_src, train_data_tar = utils.read_corpus(config.train_path)
    #dev_data_src, dev_data_tar = utils.read_corpus(config.dev_path)
    device = torch.device("cuda:0" if config.cuda else "cpu")
    model = NMT(vocab, args, device)
    model = model.to(device)
    model.train()
    optimizer = Optim(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9), config.d_model, config.warm_up_step)

    valid_num = epoch = 0
    history_valid_ppl = []
    print("begin training!", file=sys.stderr)
    while (True):
        epoch += 1
        max_iter = int(math.ceil(len(train_data)/config.train_batch_size))
        with tqdm(total=max_iter, desc="train") as pbar:
            #for batch_src, batch_tar, tar_word_num in utils.batch_iter(train_data_src, train_data_tar, config.train_batch_size):
            for batch_src, batch_tar, tar_word_num in train_loader:
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
            eval_ppl = evaluate_ppl(model, dev_data, dev_loader, config.dev_batch_size, vocab)
            print(eval_ppl)
            flag = len(history_valid_ppl) == 0 or eval_ppl < min(history_valid_ppl)
            if (flag):
                print(f"current model is the best! save to [{config.model_save_path}]", file=sys.stderr)
                history_valid_ppl.append(eval_ppl)
                model.save(os.path.join(config.model_save_path, f"01.29_{epoch}_{eval_ppl}_checkpoint.pth"))
                torch.save(optimizer.optimizer.state_dict(), os.path.join(config.model_save_path, f"01.29_{epoch}_{eval_ppl}_optimizer.optim"))
        if (epoch == config.max_epoch):
            print("reach the maximum number of epochs!", file=sys.stderr)
            return

def main():
    train()

if __name__ == '__main__':
    main()