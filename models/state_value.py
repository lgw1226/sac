import torch
import torch.nn as nn
import torch.optim as optim

from models.mlp import MLP


class StateValue():

    def __init__(
        self, ob_dim,
        num_layers=2,
        layer_size=256,
        lr=0.001,
        device=None
    ):
        
        self.ob_dim = ob_dim

        self.num_layers = num_layers
        self.layer_size = layer_size
        self.lr = lr
        self.device = device
        
        self.net = MLP(self.ob_dim, 1, self.num_layers, self.layer_size).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)

    def get_value(self, ob):

        return self.net(ob).squeeze()
    

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ob_dim = 3
    state_value = StateValue(ob_dim, device=device)
    
    # input = torch.randn(ob_dim, device=device); print(input)
    # output = state_value.get_value(input); print(output)

    batch_size = 2
    input = torch.randn((batch_size, ob_dim), device=device); print(input)
    output = state_value.get_value(input); print(output)
