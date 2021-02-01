import torch
import utils

class Vocab(object):
    
    def __init__(self, file):
        self.word2id = dict()
        word_cnt = 0
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                self.word2id[line] = word_cnt
                word_cnt += 1
            f.close()
        self.id2word = dict()
        for key, value in self.word2id.items():
            self.id2word[value] = key
    
    def __getitem__(self, word):
        return self.word2id[word]
    
    def __len__(self):
        return len(self.word2id)
    
    def __contains__(self, word):
        return word in self.word2id
    
    def id2word(self, id):
        return self.id2word[id]
    
    def sen2id(self, sents):
        '''
        sents: list[list[int]] or list[int]
        '''
        sen_id = []
        if (type(sents[0]) == list):
            for sen in sents:
                shuhe = []
                for word_id in sen:
                    shuhe.append(word_id)
                sen_id.append(shuhe)
        else:
            shuhe = []
            for word_id in sents:
                shuhe.append(word_id)
            sen_id.append(shuhe)
        return sen_id
    
    def word2tensor(self, sents, device):
        sents = self.sen2id(sents)
        sents = utils.padding(sents, self.word2id['<pad>'])
        sents_tensor = torch.tensor(sents, dtype=torch.long, device=device)
        return sents_tensor.t()

class Text(object):

    def __init__(self, src_file, tar_file):
        self.src = Vocab(src_file)
        self.tar = Vocab(tar_file)