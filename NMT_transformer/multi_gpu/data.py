import utils
from torch.utils.data import Dataset

class Data(Dataset):

    def __init__(self, src_file, tar_file):
        self.src = utils.read_corpus(src_file)
        self.tar = utils.read_corpus(tar_file, True)
        self.len_ = len(self.src)
    
    def __getitem__(self, index):
        return self.src[index], self.tar[index]
    
    def __len__(self):
        return self.len_
