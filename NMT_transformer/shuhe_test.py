import torch
import torch.nn as nn
from tqdm import tqdm
import time


'''
tensor_a = torch.BoolTensor(1, 2)
print(tensor_a.shape)
'''
'''
a = [1, 2, 3]
tensor_a = torch.tensor(a, dtype=torch.float, device=torch.device('cuda:0'))
print(tensor_a)
print((tensor_a * 2.1).dtype)
print(float('-inf'))
tensor_a[0] = float('-inf')
print(tensor_a)
for i in range(tensor_a.shape[0]):
    if (tensor_a[i] == float('-inf')):
        print(i)
print(nn.functional.softmax(tensor_a))
'''
'''
a = [[1, 2, 3], [4, 5, 6]]
tensor_a = torch.tensor(a, dtype=torch.long)
print(tensor_a)
print(tensor_a.sum(dim=0).shape)
'''
'''
path = "/home/wangshuhe/shuhelearn/ShuHeLearning/NMT_transformer/test.txt"
cnt = 0
with open(path, "r") as f:
    for line in f:
        cnt += 1
print(cnt)
'''
'''
test = {"name": "shuhe", "age": 22}
for i in tqdm(range(10), postfix=test):
    test['name'] = "wangshuhe"
    tqdm.set_postfix(test)    
    time.sleep(10)
'''
'''
test = {"name": "shuhe", "age": 22}
with tqdm(total=10, desc="test") as pbar:
    for i in range(10):
        test['age'] = i
        pbar.set_postfix(test)
        time.sleep(10)
        pbar.update(1)
'''
'''
path = "/data/wangshuhe/learn/process_data/train.txt"
shuhe = []
cnt = 0
with open(path, "r") as f:
    for line in f:
        cnt += 1
        shuhe.append(line)
        if (cnt > 100):
            break
    f.close()
path = "/data/wangshuhe/learn/process_data/tiny_tiny.txt"
with open(path, "w") as f:
    for line in shuhe:
        f.write(line)
    f.close()
'''
'''
tensor_a = torch.BoolTensor(1, 2)
print(tensor_a)
'''
'''
a = [1, 2, 3]
if (True):
    b = []
    print(a)
    print(b)
    b.append(a.copy())
    print(a)
    print(b)
    a.append(9)
    print(a)
    print(b)
    b = a.copy()
    b.append(4)
    print(a)
    print(b)
    a = b.copy()
    print(a)
    print(b)
    b.append(6)
    print(a)
    print(b)
'''
'''
a = [1, 2, 3]
tensor_a = torch.tensor(a, dtype=torch.long)
tensor_b = None
tensor_b = tensor_a
print(tensor_a)
print(tensor_b)
tensor_a = None
print(tensor_a)
print(tensor_b)
tensor_a[0] = 6
print(tensor_a)
print(tensor_b)
tensor_b[0] = 9
print(tensor_a)
print(tensor_b)
tensor_b = torch.cat((tensor_b, tensor_a), dim=-1)
print(tensor_a)
print(tensor_b)
tensor_a[1] = 10
print(tensor_a)
print(tensor_b)
'''
'''
a = [[1, 2, 3], [4, 5, 6]]
tensor_a = torch.tensor(a, dtype=torch.long)
_, index_a = torch.topk(tensor_a, 3)
print(index_a)
'''
'''
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
tensor_a = torch.tensor(a, dtype=torch.long)
_, index_a = torch.topk(tensor_a[1:5], 2)
print(index_a)
'''
path = "/data/wangshuhe/learn/process_data/test.txt"
cnt = 0
shuhe = []
with open(path, "r") as f:
    for line in f:
        shuhe.append(line)
        cnt += 1
        if (cnt >= 100):
            break
    f.close()
path = "/data/wangshuhe/learn/process_data/sub_test.txt"
with open(path, "w") as f:
    for line in shuhe:
        f.write(line)
    f.close()