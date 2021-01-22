import torch
import utils

class Vocab(object):
    '''
    Vocabulary
    '''
    def __init__(self, file):
        self.word2id = dict()
        self.word2id['<start>'] = 0
        self.word2id['<end>'] = 1
        self.word2id['<pad>'] = 2
        self.word2id['<unk>'] = 3
        self.word_offset = 2
        word_cnt = 4
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                if (line == 'Unknown'):
                    continue
                self.word2id[line] = word_cnt
                word_cnt += 1
            f.close()
        self.id2word = dict()
        for key, value in self.word2id.items():
            self.id2word[value] = key
        
    def __getitem__(self, word):
        return self.word2id[word]

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)
    
    def id2word(self, id):
        return self.id2word[id]
    
    def sen2id(self, sents):
        '''
        sents : list[int] or list[list[int]] sentence(s)
        '''
        sen_id = []
        if (type(sents[0]) == list):
            for sen in sents:
                shuhe = []
                for text_id in sen:
                    shuhe.append(text_id + self.word_offset)
                sen_id.append(shuhe)
        else:
            shuhe = []
            for text_id in sents:
                shuhe.append(text_id + self.word_offset)
            sen_id.append(shuhe)
            
        return sen_id
    
    def word2tensor(self, sents, device):
        sents_id = self.sen2id(sents)
        sents_id = utils.padding(sents_id, self['<pad>'])
        sen_tensor = torch.tensor(sents_id, dtype=torch.long, device=device)
        return sen_tensor.t()

class Text(object):
    
    def __init__(self, file_src, file_tar):
        self.src = Vocab(file_src)
        self.tar = Vocab(file_tar)
    