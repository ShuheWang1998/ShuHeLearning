import torch.nn as nn
from torch.autograd import Variable

class LabelSmoothing(nn.Module):

    def __init__(self, class_size, padding_idx, eps):
        super(LabelSmoothing, self).__init__()
        self.class_size = class_size
        self.padding_idx = padding_idx
        self.eps = eps
        self.criterion = nn.KLDivLoss(reduction='sum')
    
    def forward(self, output, target):
        '''
        output: sen_len * batch * feature
        target: sen_len * batch
        '''
        goal = output.clone()
        goal = goal.fill_(self.eps/(self.class_size-1)).scatter_(-1, index=target[1:].unsqueeze(dim=-1), value=1-self.eps)
        mask = (target != self.padding_idx).float()
        goal = goal * mask.unsqueeze(dim=-1)
        return self.criterion(output[1:].view(-1, self.class_size), Variable(goal[1:].view(-1, self.class_size), requires_grad=False))