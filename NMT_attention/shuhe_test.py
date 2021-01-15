'''
path = "/data/wangshuhe/learn/process_data/en.dict"
with open(path, "r") as f:
    cnt = 0
    for line in f:
        cnt += 1
        print(line.strip())
    print(cnt)
    f.close()
'''
import torch

'''
a = [1, 2, 3]
tensor_a = torch.tensor(a, dtype=torch.long)
print(tensor_a)
print(tensor_a[1])
'''
'''
a = [[1], [4]]
tensor_a = torch.tensor(a, dtype=torch.long)
print(tensor_a)
for i, x in enumerate(tensor_a):
    print(i)
    print(type(x.item()))
'''
'''
a = 2.1
tensor_a = torch.tensor(a, dtype=torch.float16)
print(tensor_a)
b = [1, 2, 3, 4, 5]
tensor_b = torch.tensor(b, dtype=torch.long)
print(tensor_a * tensor_b)
tensor_c = torch.zeros(5, dtype=torch.float16)
tensor_c += tensor_a * tensor_b
tensor_c += tensor_a * tensor_b
print(tensor_c)
'''
a = [[1, 2, 3], [4, 5, 6]]
tensor_a = torch.tensor(a, dtype=torch.long)
print(tensor_a)
print(tensor_a[0])
print(torch.cat((tensor_a[0].view(1, -1), tensor_a[1].view(1, -1)), dim=0).shape)
b = [1, 2, 3, 4]
tensor_b = torch.tensor(b, dtype=torch.long)
print(tensor_b)
tensor_b[0] = 6
print(tensor_b)
print(tensor_a * tensor_b[0])