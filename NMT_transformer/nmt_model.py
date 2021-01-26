import torch
import torch.nn as nn
from embeddings import Embeddings
import math

class NMT(nn.Module):

    def __init__(self, text, args, device):
        super(NMT, self).__init__()
        self.text = text
        self.args = args
        self.device = device
        self.Embeddings = Embeddings(args['embed_size'], text)
        self.transformer = nn.Transformer(d_model=args['d_model'], nhead=args['nhead'], num_encoder_layers=args['num_encoder_layers'], num_decoder_layers=args['num_decoder_layers'], dim_feedforward=args['dim_feedforward'], dropout=args['dropout'])
        self.project = nn.Linear(in_features=args['d_model'], out_features=len(self.text.tar), bias=True)
        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, source, target):
        source_tensor = self.text.src.word2tensor(source, self.device)
        target_tensor = self.text.tar.word2tensor(target, self.device)
        output = self.step(source_tensor, target_tensor)
        P = nn.functional.log_softmax(self.project(output), dim=-1)
        '''
        shuhe
        '''
        shuhe = P[:-1]
        shuhe_score = torch.zeros(shuhe.shape[0], shuhe.shape[1], dtype=torch.float, device=self.device)
        for i in range(shuhe.shape[0]):
            for j in range(shuhe.shape[1]):
                shuhe_score[i][j] = shuhe[i][j][target_tensor[i+1][j]]
        '''
        shuhe
        '''
        output_mask = (target_tensor != self.text.tar['<pad>']).float()
        score = torch.gather(P, index=target_tensor[1:].unsqueeze(dim=-1), dim=-1).squeeze(dim=-1) * output_mask[1:]
        return score.sum(dim=0)
    
    def step(self, source_tensor, target_tensor):
        target_mask = torch.BoolTensor(target_tensor.shape[0], target_tensor.shape[0])
        source_padding_mask = torch.BoolTensor(source_tensor.shape[1], source_tensor.shape[0])
        target_padding_mask = torch.BoolTensor(target_tensor.shape[1], target_tensor.shape[0])
        S = source_tensor.shape[0]
        T = target_tensor.shape[0]
        N = source_tensor.shape[1]
        for i in range(T):
            for j in range(T):
                if (j <= i):
                    target_mask[i][j] = False
                else:
                    target_mask[i][j] = True
        target_mask = target_mask.to(self.device)
        for i in range(N):
            for j in range(S):
                if (source_tensor[j][i].item() != self.text.src['<pad>']):
                    source_padding_mask[i][j] = False
                else:
                    source_padding_mask[i][j] = True
            for j in range(T):
                if (target_tensor[j][i].item() != self.text.tar['<pad>']):
                    target_padding_mask[i][j] = False
                else:
                    target_padding_mask[i][j] = True
        source_padding_mask = source_padding_mask.to(self.device)
        target_padding_mask = target_padding_mask.to(self.device)
        pre_src_PE = []
        for i in range(S):
            shuhe = []
            for j in range(self.args['embed_size']):
                if (j % 2 == 0):
                    shuhe.append(math.sin(i / math.pow(10000, j / self.args['d_model'])))
                else:
                    shuhe.append(math.cos(i / math.pow(10000, (j - 1)/self.args['d_model'])))
            pre_src_PE.append(shuhe)
        pre_tar_PE = []
        for i in range(T):
            shuhe = []
            for j in range(self.args['embed_size']):
                if (j % 2 == 0):
                    shuhe.append(math.sin(i / math.pow(10000, j / self.args['d_model'])))
                else:
                    shuhe.append(math.cos(i / math.pow(10000, (j - 1)/self.args['d_model'])))
            pre_tar_PE.append(shuhe)
        pre_src_PE = torch.tensor(pre_src_PE, dtype=torch.float, device=self.device)
        pre_tar_PE = torch.tensor(pre_tar_PE, dtype=torch.float, device=self.device)
        pre_src_PE = pre_src_PE.reshape(pre_src_PE.shape[0], 1, pre_src_PE.shape[1])
        pre_tar_PE = pre_tar_PE.reshape(pre_tar_PE.shape[0], 1, pre_tar_PE.shape[1])
        src_PE = pre_src_PE
        tar_PE = pre_tar_PE
        for i in range(N-1):
            src_PE = torch.cat((src_PE, pre_src_PE), dim=1)
            tar_PE = torch.cat((tar_PE, pre_tar_PE), dim=1)
        source_embed_tensor = self.Embeddings.src(source_tensor).to(self.device) * math.sqrt(self.args['d_model']) + self.dropout(src_PE)
        target_embed_tensor = self.Embeddings.tar(target_tensor).to(self.device) * math.sqrt(self.args['d_model']) + self.dropout(tar_PE)
        output = self.transformer(source_embed_tensor, target_embed_tensor, tgt_mask=target_mask, src_key_padding_mask=source_padding_mask, tgt_key_padding_mask=target_padding_mask)
        #output = target_embed_tensor
        return output

    def beam_search(self, source, search_size, max_tar_length):
        source_tensor = self.text.src.word2tensor(source)
        now_source_tensor = source_tensor
        now_predict = [[self.text.tar['<start>']]]
        predict = []
        now_predict_length = 0
        while (len(predict) < search_size and now_predict_length < max_tar_length):
            now_predict_length += 1
            now_predict_tensor = self.text.tar.word2tensor(now_predict)
            output = self.step(now_source_tensor, now_predict_tensor)
            output = output[now_predict_length-1]
            P = nn.functional.softmax(self.project(output), dim=-1).reshape(output.shape[0]*len(self.text.tar))
            _, topk_index = torch.topk(P, 5)
            next_predict = []
            next_source_tensor = None
            for i in range(5):
                next_word_id = topk_index[i] % len(self.text.tar)
                sent_id = topk_index[i] // len(self.text.tar)
                if (next_word_id == self.text.tar['<end>']):
                    predict.append(now_predict[sent_id][1:])
                    if (len(predict) == search_size):
                        break
                    continue
                next_predict.append(now_predict[sent_id].append(next_word_id))
                if (next_source_tensor is None):
                    next_source_tensor = source_tensor
                else:
                    next_source_tensor = torch.cat((next_source_tensor, source_tensor), dim=1)
            if (len(predict) == search_size):
                break
            if (now_predict_length == max_tar_length):
                for sen in next_predict:
                    predict.append(sen[1:])
                    if (len(predict) == search_size):
                        break
            now_source_tensor = next_source_tensor
            now_predict = next_predict
        return predict
    
    def save(self, model_path):
        params = {
            'text': self.text,
            'args': self.args,
            'device': self.device,
            'state_dict': self.state_dict()
        }
        torch.save(params, model_path)
    
    @staticmethod
    def load(model_path):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = NMT(params['text'], params['args'], params['device'])
        model.load_state_dict(params['state_dict'])
        return model