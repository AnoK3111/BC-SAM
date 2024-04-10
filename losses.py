import torch
import torch.nn as nn
import torch.nn.functional as F

class MMDLoss(nn.Module):
    def __init__(self, dataset_num):
        super(MMDLoss,self).__init__()
        self.dataset_num = dataset_num
        self.mseloss = nn.MSELoss()
    

    def forward(self, x, dataset):
        x = x.squeeze()
        loss = 0.0
        source = x[dataset == 0]
        for i in range(1, self.dataset_num):
            target = x[dataset == 1]
            loss += self.mseloss(source.mean(0), target.mean(0))
        return loss
    

if __name__=='__main__':
    c = MMDLoss(2)
    ds = torch.randint(0, 2, (128,))
    x = torch.randn((128, 8, 1, 1))
    print(c(x, ds))



