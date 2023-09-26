from copy import copy

import torch
import numpy as np

import models


class VPGAgent():

    def __init__(
        self, ob_dim, ac_dim, ac_lim,
        num_layers=2,
        layer_size=256,
        lr=0.0003,
        gamma=0.99,
        device=None
    ):

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.ac_lim = ac_lim

        self.num_layers = num_layers
        self.layer_size = layer_size
        self.lr = lr
        self.gamma = gamma
        self.device = device

        self.p = models.GaussianPolicy(self.ob_dim, self.ac_dim, self.ac_lim, num_layers=self.num_layers, layer_size=self.layer_size, lr=self.lr, device=self.device)
        self.v = models.StateValue(self.ob_dim, num_layers=num_layers, layer_size=layer_size, lr=lr, device=device)

    def rtgs(self, rwds: np.ndarray):

        rtgs = np.flip(np.cumsum(np.flip(rwds)))

        return torch.as_tensor(rtgs.copy(), dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def update_v(self, obs: torch.Tensor, rtgs: torch.Tensor):

        values = self.v.get_value(obs)

        loss = torch.mean((values - rtgs) ** 2)
        self.v.optim.zero_grad()
        loss.backward()
        self.v.optim.step()

        return loss
    
    def update_p(self, obs: torch.Tensor):

        _, logps = self.p.get_action(obs)
        values = self.v.get_value(obs)
        
        loss = -torch.mean(logps * values)
        self.p.optim.zero_grad()
        loss.backward()
        self.p.optim.step()

        return loss


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ob_dim = 3
    ac_dim = 2
    agent = VPGAgent(ob_dim, ac_dim, device)
