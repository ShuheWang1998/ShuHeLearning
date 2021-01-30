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

def get_batch(data):
    src = []
    tar = []
    tar_word_num = 0
    for sub_src, sub_tar in data:
        src.append(sub_src)
        tar.append(sub_tar)
        tar_word_num += len(sub_tar)
    return src, tar, tar_word_num

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

'''
def batch_iter(data_src, data_tar, batch_size):
    src_sents = []
    tar_sents = []
    tar_batch_len = 0
    len_ = len(data_src)
    for i in range(len_):
        src_sents.append(data_src[i])
        tar_sents.append(data_tar[i])
        tar_batch_len += len(data_tar[i]) - 1
        if ((i + 1) % batch_size == 0):
            yield src_sents, tar_sents, tar_batch_len
            tar_batch_len = 0
            src_sents = []
            tar_sents = []
    if (len_ % batch_size != 0):
        yield src_sents, tar_sents, tar_batch_len

def get_num(file_path):
    cnt = 0
    with open(file_path, "r") as f:
        for line in f:
            cnt += 1
        f.close()
    return cnt
'''