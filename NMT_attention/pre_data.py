word_src = {}
word_src['<start>'] = 0
word_src['<end>'] = 1
word_src['<pad>'] = 2
word_src['<unk>'] = 3
word_src_cnt = 4
word_tar = {}
word_tar['<start>'] = 0
word_tar['<end>'] = 1
word_tar['<pad>'] = 2
word_tar['<unk>'] = 3
word_tar_cnt = 4
src_input = "/data/wangshuhe/learn/process_data/de.dict"
tar_input = "/data/wangshuhe/learn/process_data/en.dict"
src_output = "/data/wangshuhe/learn/process_data/LSTM/de.txt"
tar_output = "/data/wangshuhe/learn/process_data/LSTM/en.txt"
src2new = {}
tar2new = {}
src_cnt = 0
tar_cnt = 0

with open(src_input, "r") as f:
    for line in f:
        src_cnt += 1
        line = line.strip()
        if (line not in word_src):
            if (line != 'Unknown'):
                word_src[line] = word_src_cnt
                word_src_cnt += 1
        if (line == 'Unknown'):
            src2new[src_cnt] = 3
        else:
            src2new[src_cnt] = word_src[line]
    f.close()

with open(src_output, "w") as f:
    for key, value in word_src.items():
        f.write(key+'\n')
    f.close()

with open(tar_input, "r") as f:
    for line in f:
        tar_cnt += 1
        line = line.strip()
        if (line not in word_tar):
            if (line != 'Unknown'):
                word_tar[line] = word_tar_cnt
                word_tar_cnt += 1
        if (line == 'Unknown'):
            tar2new[tar_cnt] = 3
        else:
            tar2new[tar_cnt] = word_tar[line]
    f.close()

with open(tar_output, "w") as f:
    for key, value in word_tar.items():
        f.write(key+'\n')
    f.close()

print(src_cnt, tar_cnt)

for now_path in ['train', 'dev', 'test']:
    sen_input = f"/data/wangshuhe/learn/process_data/{now_path}.txt"
    src_output = f"/data/wangshuhe/learn/process_data/LSTM/only_de_sen_{now_path}.txt"
    tar_output = f"/data/wangshuhe/learn/process_data/LSTM/only_en_sen_{now_path}.txt"
    src_sen = []
    tar_sen = []
    print(sen_input)
    with open(sen_input, "r") as f:
        for line in f:
            line = line.strip().split()
            now = []
            for word in line:
                data = 0
                for ch in word:
                    if (ch == '|'):
                        now.append(data)
                        src_sen.append(now)
                        now = []
                        data = 0
                        continue
                    data = data * 10 + int(ch)
                now.append(data)
            tar_sen.append(now)
        f.close()

    with open(src_output, "w") as f:
        for sen in src_sen:
            now_str = ""
            for word_id in sen:
                now_str += str(src2new[word_id]) + " "
            now_str += '\n'
            f.write(now_str)
        f.close()
    
    with open(tar_output, "w") as f:
        for sen in tar_sen:
            now_str = ""
            for word_id in sen:
                now_str += str(tar2new[word_id]) + " "
            now_str += '\n'
            f.write(now_str)
        f.close()