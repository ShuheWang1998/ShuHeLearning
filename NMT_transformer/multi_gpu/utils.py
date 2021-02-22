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

def read_corpus(file_path, flag=False):
    output = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip().split()
            now = []
            if (flag):
                now.append(0)
            for word in line:
                now.append(int(word))
            if (flag):
                now.append(1)
            output.append(now)
    return output

def get_batch(data):
    src = []
    tar = []
    tar_word_num = 0
    for sub_src, sub_tar in data:
        src.append(sub_src)
        tar.append(sub_tar)
        tar_word_num += len(sub_tar)
    return src, tar, tar_word_num

'''
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
'''