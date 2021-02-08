import math
import torch

class Optim():
    '''
    def __init__(self, optimizer, d_model, warm_up_step):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warm_up_step = warm_up_step
        self.n_current_step = 0
        self.init_lr = math.pow(self.d_model, -0.5)

    def step_and_updata_lr(self):
        self.updata_lr()
        self.optimizer.step()

    def get_lr(self):
        return min(math.pow(self.n_current_step, -0.5), math.pow(self.warm_up_step, -1.5)*self.n_current_step)
    
    def updata_lr(self):
        self.n_current_step += 1
        lr = self.init_lr * self.get_lr()
        for para in self.optimizer.param_groups:
            para['lr'] = lr
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    '''
    def __init__(self, optimizer, warm_up_step, init_lr, end_lr):
        self.warm_up_step = warm_up_step
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.lrs = torch.linspace(self.init_lr, self.end_lr, self.warm_up_step)
        self.lr = self.lrs[0].item()
        for para in self.optimizer.param_groups:
            para['lr'] = self.lr
        self.update_num = 0
    
    def step(self):
        self.optimizer.step()
    
    def updata_lr(self):
        self.update_num += 1
        if (self.update_num < self.warm_up_step):
            self.lr = self.lrs[self.update_num].item()
        else:
            decay_factor = self.end_lr * math.pow(self.warm_up_step, 0.5)
            self.lr = decay_factor / math.pow(self.update_num, 0.5)
        for para in self.optimizer.param_groups:
            para['lr'] = self.lr
    
    def zero_grad(self):
        self.optimizer.zero_grad()