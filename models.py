import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class shallowDQN(nn.Module):
    def __inti__(self, outputs):
        super(ShallowDQN,self).__init__()

        self.lin1 = nn.Linear(9,15)
        self.lin2 = nn.Linear(15,9)
        self.lin3 = nn.Linear(9,3)

    def forward(self,x):
        x = x.to(device)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x.view(x.size(0),-1)


class DeepDQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()
        
        self.lin1 = nn.Linear(9,80)
        self.lin2 = nn.Linear(80,50)
        self.lin3 = nn.Linear(50,25)
        self.lin4 = nn.Linear(25,15)
        self.lin5 = nn.Linear(15,8)
        self.lin6 = nn.Linear(8,3)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = self.lin6(x)
        
        return x.view(x.size(0), -1)
