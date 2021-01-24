def padding(sents, pad_word):
    '''
    sents: list[list[int]]
    '''
    max_ = 0
    for sen in sents:
        max_ = max(max_, len(sen))
    padding_sents = []
    for sen in sents:
        while (len(sen) < max_):
            sen.append(pad_word)
        padding_sents.append(sen)
    return padding_sents