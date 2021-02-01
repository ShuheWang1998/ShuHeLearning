import torch
import torch.nn as nn
import math
import shuhe_config as config

class NMT(nn.Module):

    def __init__(self, vocab, args, device):
        super(NMT, self).__init__()
        self.vocab = vocab
        self.args = args
        self.device = device
        self.Embeddings = nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=args['embed_size'], padding_idx=self.vocab['<pad>'])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args['d_model'], nhead=args['nhead'], dim_feedforward=args['dim_feedforward'], dropout=args['dropout'])
        self.encoder_norm = nn.LayerNorm(args['d_model'])
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=args['num_encoder_layers'], norm=self.encoder_norm)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=args['d_model'], nhead=args['nhead'], dim_feedforward=args['dim_feedforward'], dropout=args['dropout'])
        self.decoder_norm = nn.LayerNorm(args['d_model'])
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=args['num_decoder_layers'], norm=self.decoder_norm)
        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, source, target):
        source_tensor = self.vocab.word2tensor(source, self.device)
        target_tensor = self.vocab.word2tensor(target, self.device)
        memory, memory_padding_mask = self.encode(source_tensor)
        output = self.decode(memory, memory_padding_mask, target_tensor)
        P = nn.functional.log_softmax(nn.functional.linear(input=output, weight=self.Embeddings.weight), dim=-1)
        output_mask = (target_tensor != self.vocab['<pad>']).float()
        score = torch.gather(P, index=target_tensor[1:].unsqueeze(dim=-1), dim=-1).squeeze(dim=-1) * output_mask[1:]
        return score.sum(dim=0)

    def encode(self, source_tensor):
        S = source_tensor.shape[0]
        N = source_tensor.shape[1]
        source_padding_mask = (source_tensor == self.vocab['<pad>']).bool().t()
        source_padding_mask = source_padding_mask.to(self.device)
        source_embed_tensor = self.Embeddings(source_tensor).to(self.device) * math.sqrt(self.args['d_model']) + self.dropout(self.get_position(S, N))
        output = self.encoder(source_embed_tensor, src_key_padding_mask=source_padding_mask)
        # output: sen_len * batch_size * feature_size
        # source_padding_mask: batch_size * sen_len
        return output, source_padding_mask
    
    def decode(self, memory, memory_padding_mask, target_tensor):
        target_mask = torch.BoolTensor(target_tensor.shape[0], target_tensor.shape[0])
        T = target_tensor.shape[0]
        N = target_tensor.shape[1]
        for i in range(T-1):
            target_mask[i][:i+1] = False
            target_mask[i][i+1:] = True
        target_mask[T-1][0:] = False
        target_mask = target_mask.to(self.device)
        target_padding_mask = (target_tensor == self.vocab['<pad>']).bool().t()
        target_padding_mask = target_padding_mask.to(self.device)
        target_embed_tensor = self.Embeddings(target_tensor).to(self.device) * math.sqrt(self.args['d_model']) + self.dropout(self.get_position(T, N))
        output = self.decoder(target_embed_tensor, memory, tgt_mask=target_mask, tgt_key_padding_mask=target_padding_mask, memory_key_padding_mask=memory_padding_mask)
        # output: sen_len * batch_size * feature
        return output

    def get_position(self, sen_len, batch_size):
        pre_PE = []
        for i in range(sen_len):
            shuhe = []
            for j in range(self.args['embed_size']):
                if (j % 2 == 0):
                    shuhe.append(math.sin(i / math.pow(10000, j / self.args['d_model'])))
                else:
                    shuhe.append(math.cos(i / math.pow(10000, (j - 1)/self.args['d_model'])))
            pre_PE.append(shuhe)
        pre_PE = torch.tensor(pre_PE, dtype=torch.float, device=self.device)
        pre_PE = pre_PE.reshape(pre_PE.shape[0], 1, pre_PE.shape[1])
        pre_PE = pre_PE.expand(pre_PE.shape[0], batch_size, pre_PE.shape[2])
        return pre_PE

    def beam_search(self, source, search_size, max_tar_length, batch_size):
        '''
        source_tensor = self.text.src.word2tensor(source, self.device)
        now_source_tensor = source_tensor
        now_predict = [[0]]
        predict = []
        now_predict_length = 0
        now_score = torch.zeros(1, dtype=torch.float, device=self.device).reshape(1, 1)
        while (now_predict_length < max_tar_length):
            now_predict_length += 1
            now_predict_tensor = self.text.tar.word2tensor(now_predict, self.device)
            output = self.step(now_source_tensor, now_predict_tensor)[now_predict_length-1]
            P = (nn.functional.softmax(self.project(output), dim=-1)+now_score).reshape(output.shape[0]*len(self.text.tar))
            score, topk_index = torch.topk(P, 5)
            score = score.cuda()
            topk_index = topk_index.cuda()
            next_predict = []
            next_source_tensor = None
            next_score = []
            for i in range(5):
                next_word_id = topk_index[i].item() % len(self.text.tar)
                sent_id = topk_index[i].item() // len(self.text.tar)
                if (next_word_id == self.text.tar['<end>']):
                    predict.append(now_predict[sent_id][1:].copy())
                    if (len(predict) == search_size):
                        break
                    continue
                next_predict.append(now_predict[sent_id].copy())
                next_predict[-1].append(next_word_id)
                next_score.append(score[i].item())
                if (next_source_tensor is None):
                    next_source_tensor = source_tensor
                else:
                    next_source_tensor = torch.cat((next_source_tensor, source_tensor), dim=1)
            if (len(predict) == search_size):
                break
            if (now_predict_length == max_tar_length):
                for sen in next_predict:
                    predict.append(sen[1:].copy())
                    if (len(predict) == search_size):
                        break
                break
            now_score = torch.tensor(next_score, dtype=torch.float, device=self.device).reshape(-1, 1)
            now_source_tensor = next_source_tensor
            now_predict = next_predict.copy()
        return predict
        '''
        source_tensor = self.vocab.word2tensor(source, self.device)
        memory, memory_padding = self.encode(source_tensor)
        now_memory = memory
        now_memory_padding = memory_padding
        now_predict = [[0] for _ in range(batch_size)]
        predict = [[] for _ in range(batch_size)]
        now_predict_length = 0
        now_score = torch.zeros(batch_size, dtype=torch.float, device=self.device).reshape(batch_size, 1)
        batch_index = [(i, 1) for i in range(batch_size)]
        while (now_predict_length < max_tar_length):
            now_predict_length += 1
            now_predict_tensor = self.vocab.word2tensor(now_predict, self.device)
            output = self.decode(now_memory, now_memory_padding, now_predict_tensor)[-1]
            P = (nn.functional.log_softmax(nn.functional.linear(output, self.Embeddings.weight), dim=-1)+now_score).reshape(output.shape[0]*len(self.vocab))
            now_memory = memory.permute(1, 0, 2)
            now_memory_padding = memory_padding
            next_memory = None
            next_memory_padding = None
            next_batch_index = []
            now_start = 0
            next_predict = []
            next_score = []
            flag = False
            for key, value in batch_index:
                score, topk_index = torch.topk(P[len(self.vocab)*now_start:len(self.vocab)*(value+now_start)], search_size)
                next_value = 0
                now_flag = False
                for i in range(search_size):
                    next_word_id = topk_index[i].item() % len(self.vocab)
                    sent_id = topk_index[i].item() // len(self.vocab)
                    if (next_word_id == self.vocab['<end>']):
                        if (len(now_predict[now_start+sent_id][1:]) == 0):
                            continue
                        predict[key].append(((score[i].item()-now_score[now_start][0].item())/math.pow(len(now_predict[now_start+sent_id][1:]), config.alpha), now_predict[now_start+sent_id][1:].copy()))
                        if (len(predict[key]) == search_size):
                            now_flag = True
                            break
                        continue
                now_start += value
                if (now_flag):
                    continue
                for i in range(search_size):
                    next_word_id = topk_index[i].item() % len(self.vocab)
                    sent_id = topk_index[i].item() // len(self.vocab)
                    if (next_word_id == self.vocab['<end>']):
                        continue
                    if (now_predict_length == max_tar_length):
                        predict[key].append((score[i].item()/math.pow(len(now_predict[now_start-value+sent_id][1:])+1, config.alpha), now_predict[now_start-value+sent_id][1:].copy()))
                        predict[key][-1][1].append(next_word_id)
                        if (len(predict[key]) == search_size):
                            now_flag = True
                            break
                        continue
                    next_value += 1
                    next_predict.append(now_predict[now_start-value+sent_id].copy())
                    next_predict[-1].append(next_word_id)
                    next_score.append(score[i].item())
                    if (next_memory is None):
                        next_memory = now_memory[key].unsqueeze(dim=0)
                        next_memory_padding = now_memory_padding[key].unsqueeze(dim=0)
                    else:
                        next_memory = torch.cat((next_memory, now_memory[key].unsqueeze(dim=0)), dim=0)
                        next_memory_padding = torch.cat((next_memory_padding, now_memory_padding[key].unsqueeze(dim=0)), dim=0)
                if (now_flag):
                    continue
                flag = True
                next_batch_index.append((key, next_value))
            if (not flag):
                break
            now_score = torch.tensor(next_score, dtype=torch.float, device=self.device).reshape(-1, 1)
            now_memory = next_memory.permute(1, 0, 2)
            now_memory_padding = next_memory_padding
            now_predict = next_predict
            batch_index = next_batch_index
        output = []
        for sub in predict:
            sub = sorted(sub, key=lambda sc: sc[0], reverse=True)
            output.append(sub[0][1])
        return output
        
    def save(self, model_path):
        params = {
            'vocab': self.vocab,
            'args': self.args,
            'device': self.device,
            'state_dict': self.state_dict()
        }
        torch.save(params, model_path)
    
    @staticmethod
    def load(model_path):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = NMT(params['vocab'], params['args'], params['device'])
        model.load_state_dict(params['state_dict'])
        return model