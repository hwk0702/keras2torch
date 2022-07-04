import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out
    
    
class ConditionalLinearModel_shallow(nn.Module):
    def __init__(self, n_steps):
        super().__init__()
        self.lin1 = ConditionalLinear(28*28, 28*28*2, n_steps)
        self.lin2 = ConditionalLinear(28*28*2, 28*28*2, n_steps)
        self.lin3 = nn.Linear(28*28*2, 28*28)
    
    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x, y))
        x = F.relu(self.lin2(x, y))
        x = self.lin3(x)
        x = x.view(-1, 28, 28)
        return x
    

class ConditionalLinearModel_deep(nn.Module):
    def __init__(self, n_steps):
        super().__init__()
        self.lin1 = ConditionalLinear(28*28, 28*28*2, n_steps)
        self.dr1 = nn.Dropout(p=0.5)
        self.lin2 = ConditionalLinear(28*28*2, 28*28*4, n_steps)
        self.lin3 = ConditionalLinear(28*28*4, 28*28*6, n_steps)
        self.dr3 = nn.Dropout(p=0.5)
        self.lin4 = ConditionalLinear(28*28*6, 28*28*8, n_steps)
        self.lin5 = ConditionalLinear(28*28*8, 28*28*6, n_steps)
        self.dr5 = nn.Dropout(p=0.5)
        self.lin6 = ConditionalLinear(28*28*6, 28*28*4, n_steps)
        self.lin7 = ConditionalLinear(28*28*4, 28*28*2, n_steps)
        self.dr7 = nn.Dropout(p=0.5)
        self.lin8 = nn.Linear(28*28*2, 28*28)
    
    def forward(self, x, y):
        x = x.view(x.size(0), -1)  # flatten x for MLP
        x = F.relu(self.lin1(x, y))
        x = self.dr1(x)
        x = F.relu(self.lin2(x, y))
        x = F.relu(self.lin3(x, y))
        x = self.dr3(x)
        x = F.relu(self.lin4(x, y))
        x = F.relu(self.lin5(x, y))
        x = self.dr5(x)
        x = F.relu(self.lin6(x, y))
        x = F.relu(self.lin7(x, y))
        x = self.dr7(x)
        x = self.lin8(x)
        x = x.view(-1, 28, 28)
        
        return x