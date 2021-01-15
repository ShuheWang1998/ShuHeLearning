import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from embeddings import Embeddings


class NMT(nn.Module):

    def __init__(self, text, options, device):
        super(NMT, self).__init__()
        self.options = options
        self.embeddings = Embeddings(options.embed_size, text)
        self.hidden_size = options.hidden_size
        self.window_size_d = options.window_size_d
        self.text = text
        self.device = device
        self.encoder_layer = options.encoder_layer
        self.decoder_layers = options.decoder_layers

        self.encoder = nn.LSTM(num_layers=options.encoder_layer, bias=True, dropout=options.dropout_rate, bidirectional=False)
        self.decoder = nn.LSTM(num_layers=options.decoder_layers, bias=True, dropout=options.dropout_rate, bidirectional=False)
        self.ht2tan = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.tan2pt = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)
        self.ct2ht = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size, bias=False)
        self.ht2final = nn.Linear(in_features=self.hidden_size, out_features=len(self.text.tar), bias=False)
        self.dropout = nn.Dropout(options.dropout_rate)
    
    def forward(self, source: list[list[int]], target: list[list[int]]) -> torch.Tensor:
        len_ = 0
        for sen in source:
            len_ = max(len_, len(sen))
        source_tensor = self.text.src.word2tensor(source, self.device)
        target_tensor = self.text.tar.word2tensor(target, self.device)
        encode_h, encode_len, encode_hn_cn = self.encode(source_tensor, len_)
        encode_mask = self.generate_mask(encode_h, encode_len)
        

    def encode(self, source_tensor, source_length):
        x = self.embeddings.src(source_tensor)
        x = pack_padded_sequence(x, source_length)
        output, (hn, cn) = self.encoder(x)
        output, each_len = pad_packed_sequence(output)
        output = output.permute(1, 0, 2)
        return output, each_len, (hn, cn)

    def decode(self, h0_c0, encode_h, encode_len, target_tensor):
        target_tensor = target_tensor[:-1]
        y = self.embeddings.tar(target_tensor)
        ht_ct = h0_c0
        ht = torch.zeros(encode_h.shape[0], self.hidden_size, device=self.device)
        output = []
        for y_t in y:
            now_ht_ct, now_ht = self.step(encode_h, encode_len, torch.cat((y_t, ht), dim=1), y_t, ht_ct)
            output.append(now_ht)
            ht_ct = now_ht_ct
            ht = now_ht
        return torch.stack(output).to(self.device)

    def step(self, encode_h, encode_len, pre_yt, pre_ht_ct):
        yt, ht_ct = self.decoder(pre_yt, pre_ht_ct)
        yt = torch.squeeze(yt, dim=0)
        pt = nn.functional.sigmoid(self.tan2pt(nn.functional.tanh(self.ht2tan(yt))))
        batch_ct = None
        for i, each_pt in enumerate(pt):
            each_pt = encode_len[i].item() * each_pt.item()
            left = max(0, int(each_pt) - self.window_size_d)
            right = min(encode_len[i].item(), int(each_pt) + self.window_size_d)
            align = None
            for j in range(left, right):
                if (j == left):
                    align = encode_h[i][j].view(1, -1)
                else:
                    align = torch.cat((align, encode_h[i][h].view(1, -1)), dim=0)
            align = nn.functional.softmax(torch.squeeze(torch.bmm(yt[1].view(1, -1), align), dim=0))
            ex_p = torch.zeros(right-left, dtype=torch.float16)
            for j in range(left, right):
                ex_p[j-left] = math.exp(-(j-each_pt)*(j-each_pt)/(self.window_size_d*self.window_size_d/2))
            ex_p.to(self.device)
            at = align * ex_p
            ct = torch.zeros(self.hidden_size, dtype=torch.float16)
            ct.to(self.device)
            for j in range(left, right):
                ct += at[j-left]*encode_h[i][j]
            if (i == 0):
                batch_ct = ct.view(1, -1)
            else:
                batch_ct = torch.cat((batch_ct, ct.view(1, -1)), dim=0)
        ht = self.ct2ht(batch_ct)
        return ht_ct, ht


    def generate_mask(self, sen_tensor, each_len):
        mask = torch.zeros(sen_tensor.shape[0], sen_tensor.shape[1], dtype=torch.long)
        for i, j in enumerate(each_len):
            mask[i, j:] = 1
        return mask.to(self.device)