import torch
import torch.nn as nn
import torch.optim as optim


class StateActionValue():

    def __init__(self, ob_dim, ac_dim, lr=0.001, device=None):
        
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

        self.lr = lr
        self.device = device
        
        self.net = StateActionValueNet(self.ob_dim + self.ac_dim).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def get_value(self, ob, ac):

        input = torch.cat((ob, ac), dim=1)

        return self.net(input).squeeze()
    
    def _update(self, ob, ac, value_target):

        loss = self.criterion(self.get_value(ob, ac), value_target)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss


class StateActionValueNet(nn.Module):

    def __init__(self, in_dim):

        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):

        return self.fc(x)
    

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ob_dim = 3
    ac_dim = 2
    state_value = StateActionValue(ob_dim, ac_dim, device)
    
    # input = torch.randn(ob_dim, device=device); print(input)
    # output = state_value.get_value(input); print(output)

    batch_size = 2
    ob = torch.randn((batch_size, ob_dim), device=device); print(ob)
    ac = torch.randn((batch_size, ac_dim), device=device); print(ac)
    cat = torch.cat((ob, ac), dim=1); print(cat)
    output = state_value.get_value(ob, ac); print(output)

    target = torch.randn(batch_size, device=device); print(target)

    for _ in range(100):
        state_value._update(ob, ac, target)
    
    output = state_value.get_value(ob, ac); print(output)
