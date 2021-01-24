import torch
import torch.nn as nn
import shuhe_config as config
from embeddings import Embeddings
import math

class NMT(nn.Module):

    def __init__(self, text, device):
        super(NMT, self).__init__()
        self.text = text
        self.device = device
        self.Embeddings = Embeddings(config.embed_size, text)
        self.transformer = nn.Transformer(d_model=config.d_model, nhead=config.nhead, num_encoder_layers=config.num_encoder_layers, num_decoder_layers=config.num_decoder_layers, dim_feedforward=config.dim_feedforward, dropout=config.dropout)
        self.project = nn.Linear(in_features=)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, source, target):
        source_tensor = self.text.src.word2tensor(source)
        target_tensor = self.text.tar.word2tensor(target)
        target_mask = torch.BoolTensor(target_tensor.shape[0], target_tensor.shape[0])
        source_padding_mask = torch.BoolTensor(source_tensor.shape[1], source_tensor.shape[0])
        target_padding_mask = torch.BoolTensor(target_tensor.shape[1], target_tensor.shape[0])
        S = len(source_tensor.shape[0])
        T = len(target_tensor.shape[0])
        N = len(source_tensor.shape[1])
        for i in range(T):
            for j in range(0, i + 1):
                target_mask[i][j] = False
        target_mask = target_mask.to(self.device)
        for i in range(N):
            for j in range(S):
                if (source_tensor[i][j].item() == self.text.src['<pad>']):
                    source_padding_mask[i][j] = False
            for j in range(T):
                if (target_tensor[i][j].item() == self.text.tar['<pad>']):
                    target_padding_mask[i][j] = False
        source_padding_mask = source_padding_mask.to(self.device)
        target_padding_mask = target_padding_mask.to(self.device)
        pre_src_PE = []
        for i in range(S):
            shuhe = []
            for j in range(config.embed_size):
                if (j % 2 == 0):
                    shuhe.append(math.sin(i / math.pow(10000, j / config.d_model)))
                else:
                    shuhe.append(math.cos(i / math.pow(10000, (j - 1)/config.d_model)))
            pre_src_PE.append(shuhe)
        pre_tar_PE = []
        for i in range(T):
            shuhe = []
            for j in range(config.embed_size):
                if (j % 2 == 0):
                    shuhe.append(math.sin(i / math.pow(10000, j / config.d_model)))
                else:
                    shuhe.append(math.cos(i / math.pow(10000, (j - 1)/config.d_model)))
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
        source_embed_tensor = self.Embeddings.src(source_tensor).to(self.device) + src_PE
        target_embed_tensor = self.Embeddings.tar(target_tensor).to(self.device) + tar_PE
        output = self.transformer(source_embed_tensor, target_embed_tensor, tgt_mask=target_mask, src_key_padding_mask=source_padding_mask, tgt_key_padding_mask=target_padding_mask)
        
        