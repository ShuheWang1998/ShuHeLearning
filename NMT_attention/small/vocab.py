import torch
import utils

class Vocab(object):
    '''
    Vocabulary
    '''
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
                    shuhe.append(text_id)
                sen_id.append(shuhe)
        else:
            shuhe = []
            for text_id in sents:
                shuhe.append(text_id)
            sen_id.append(shuhe)
            
        return sen_id
    
    def word2tensor(self, sents, device):
        sents_id = self.sen2id(sents)
        sents_id = utils.padding(sents_id, self['<pad>'])
        sen_tensor = torch.tensor(sents_id, dtype=torch.long, device=device).cuda()
        return sen_tensor.t()

class Text(object):

    def __init__(self, source, target):
        self.src = Vocab(source)
        self.tar = Vocab(target)