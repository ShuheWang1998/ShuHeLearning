from vocab import Text

def padding(sents, pad_word):
    '''
    sents : list[list[int]] sentences
    '''
    max_ = 0
    for sent in sents:
        max_ = max(max_, len(sent))
    padding_sents = []
    for sent in sents:
        while (len(sent) < max_):
            sent.append(pad_word)
        padding_sents.append(sent)
    
    return padding_sents

def get_vocab(file_src, file_tar):
    return Text(file_src, file_tar)