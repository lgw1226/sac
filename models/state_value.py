import torch
import torch.nn as nn
import torch.optim as optim


class StateValue():

    def __init__(self, ob_dim, lr=0.001, device=None):
        
        self.ob_dim = ob_dim

        self.lr = lr
        self.device = device
        
        self.net = StateValueNet(self.ob_dim).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def get_value(self, ob):

        return self.net(ob).squeeze()
    
    def _update(self, ob, value_target):

        loss = self.criterion(self.get_value(ob), value_target)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss

class StateValueNet(nn.Module):

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
    state_value = StateValue(ob_dim, device=device)
    
    # input = torch.randn(ob_dim, device=device); print(input)
    # output = state_value.get_value(input); print(output)

    batch_size = 2
    input = torch.randn((batch_size, ob_dim), device=device); print(input)
    output = state_value.get_value(input); print(output)

    target = torch.randn(batch_size, device=device); print(target)

    for _ in range(100):
        state_value._update(input, target)
    
    output = state_value.get_value(input); print(output)
