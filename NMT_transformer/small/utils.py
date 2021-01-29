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

def read_corpus(file_path):
    src = []
    tar = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip().split()
            now = []
            for word in line:
                data = 0
                for ch in word:
                    if (ch == '|'):
                        now.append(data+2)
                        src.append(now)
                        now = [0]
                        data = 0
                        continue
                    data = data * 10 + int(ch)
                now.append(data+2)
            now.append(1)
            tar.append(now)
    return src, tar

def get_num(file_path):
    cnt = 0
    with open(file_path, "r") as f:
        for line in f:
            cnt += 1
    return cnt

def batch_iter(source, target, batch_size):
    src = []
    tar = []
    len_ = len(source)
    tar_batch_len = 0
    for i in range(len_):
        src.append(source[i])
        tar.append(target[i])
        tar_batch_len += len(target[i]) - 1
        if ((i+1) % batch_size == 0):
            yield src, tar, tar_batch_len
            tar_batch_len = 0
            src = []
            tar = []
    if (len_ % batch_size != 0):
        yield src, tar, tar_batch_len