class Optim():
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr = 0.001
        for para in self.optimizer.param_groups:
            para['lr'] = self.lr
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step_and_updata_lr(self):
        self.optimizer.step()

    def updata_lr(self):
        self.lr = self.lr / 2
        for para in self.optimizer.param_groups:
            para['lr'] = self.lr