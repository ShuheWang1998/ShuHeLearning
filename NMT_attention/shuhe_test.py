
path = "/data/wangshuhe/learn/process_data/en.dict"
with open(path, "r") as f:
    cnt = 0
    for line in f:
        cnt += 1
        print(line.strip())
    print(cnt)
    f.close()