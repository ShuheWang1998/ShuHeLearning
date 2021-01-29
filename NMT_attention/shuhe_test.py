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
import collections

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
'''
'''
a = [1, 2, 3]
tensor_a = torch.tensor(a, dtype=torch.long)
b = [4, 5, 6]
tensor_b = torch.tensor(b, dtype=torch.long)
print(tensor_a.view(1, -1))
print(tensor_b.view(1, -1))
print(torch.cat((tensor_a.view(1, -1), tensor_b.view(1, -1)), dim=1).shape)
'''
'''
a = [[1, 2, 3], [4, 5, 6]]
tensor_a = torch.tensor(a, dtype=torch.long)
print(tensor_a)
tensor_b = (tensor_a != 2).float()
print(tensor_b)
print(tensor_a.sum(dim=0))
'''
'''
a = "1"
print(type(int(a[0])))
'''
'''
a = [1, 2, 3]
b = [4, 5, 6, 7]

print(list(zip(a, b)))

def test():
    for i in range(len(a)):
        yield a[i], b[i]

for item_a, item_b in test():
    print(item_a, item_b)
'''
'''
a = [[1, 2, 3], [4, 5, 6]]
print(a)
for test in a:
    test[0] = 9
print(a)
'''
'''
Test = collections.namedtuple("shuhe", ["age", "sex", "name"])
test = Test(12, "f", "shuhe")
print(test)
'''
'''
a = [1, 2, 3]
tensor_a = torch.tensor(a, dtype=torch.long)
print(torch.topk(tensor_a, 3))
value, index = torch.topk(tensor_a, 3)
print(value)
print(index)
'''
'''
a = [[1, 2, 3], [4, 5, 6]]
tensor_a = torch.tensor(a)
print(tensor_a[1])
'''
'''
a = [1, 2, 3]
tensor_a = torch.tensor(a, dtype=torch.long)
b = [4, 5, 6]
tensor_b = torch.tensor(b, dtype=torch.float)
print(torch.cat((tensor_a, tensor_b), dim=0))
'''
'''
a = [[1, 2, 3], [4, 5, 6]]
tensor_a = torch.tensor(a, dtype=torch.long)
for test in tensor_a:
    print(test)
'''
'''
a = [1, 2, 3, 4]
tensor_a = torch.tensor(a, dtype=torch.long)
for i in range(4):
    print(tensor_a[i].item())
    print(type(tensor_a[i].item()))
'''
'''
a = [1, 2, 3]
for i in range(3):
    a[i] = "sd"
print(a)
'''
'''
a = 1
tensor_a = torch.tensor(a, dtype=torch.int64)
print(tensor_a)
'''
'''
a = [[[1, 2, 3]]]
tensor_a = torch.tensor(a, dtype=torch.long)
print(tensor_a.shape)
print(torch.squeeze(tensor_a, dim=0).squeeze(dim=0).shape)
'''
'''
a = [1, 2, 3]
tensor_a = torch.tensor(a, dtype=torch.float)
b = [4, 5, 6]
tensor_b = torch.tensor(b, dtype=torch.float)
print(tensor_a)
print(tensor_a.shape)
print(tensor_b)
print(tensor_b.shape)
print(tensor_a * tensor_b)
print((tensor_a * tensor_b).shape)
tensor_a[4:] = 9
print(tensor_a)
tensor_a = torch.arange(0, 8, 1, dtype=torch.long)
print(tensor_a)
'''
'''
a = [[1, 3, 5], [2, 4, 6]]
tensor_a = torch.tensor(a, dtype=torch.long)
b = [8, 9]
tensor_b = torch.tensor(b, dtype=torch.long).reshape(2, 1)
print(tensor_a-tensor_b)
print(-tensor_a)
a = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
tensor_a = torch.tensor(a, dtype=torch.float)
b = [[1, 2], [3, 4]]
tensor_b = torch.tensor(b, dtype=torch.float)
print(tensor_a.shape)
print(tensor_b.shape)
print(tensor_a)
print(tensor_a * tensor_b.reshape(tensor_b.shape[0], tensor_b.shape[1], 1))
'''
tensor_a = torch.tensor([1], dtype=torch.float)
tensor_b = torch.tensor([float('-inf')], dtype=torch.float)
print(tensor_a+tensor_b)