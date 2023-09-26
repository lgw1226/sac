from math import log

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F

from models.mlp import MLP


LOG_STD_MIN = -10
LOG_STD_MAX = 2


# squashed gaussian policy (numerically more stable than standard gaussian policy)
class SquashedGaussianPolicy():

    def __init__(
        self, ob_dim, ac_dim, ac_lim,
        num_layers=2,
        layer_size=256,
        lr=0.001,
        device=None
    ):
        
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.ac_lim = ac_lim

        self.num_layers = num_layers
        self.layer_size = layer_size
        self.lr = lr
        self.device = device
        
        self.net = MLP(self.ob_dim, self.ac_dim * 2, self.num_layers, self.layer_size).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)

    def get_dist(self, ob):

        output = self.net(ob)

        mean = output[:,:self.ac_dim]
        log_std = torch.clip(output[:,self.ac_dim:], LOG_STD_MIN, LOG_STD_MAX)

        dist = distributions.Normal(mean, torch.exp(log_std))

        return dist
    
    def get_action(self, ob, eval=False):

        if not eval:
            dist = self.get_dist(ob)
            action = dist.rsample()
        else:
            dist = self.get_dist(ob)
            action = dist.loc

        tanh_action = torch.tanh(action) * self.ac_lim
        logp = dist.log_prob(action)
        logp -= 2 * (log(2) - action - F.softplus(-2 * action))
        
        return tanh_action, logp.sum(-1, keepdim=True)
    

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ob_dim = 3
    ac_dim = 1
    policy = SquashedGaussianPolicy(ob_dim, ac_dim, 1, device=device)

    batch_size = 4
    ob = torch.randn((batch_size, ob_dim), device=device); print(ob)
    dist = policy.get_dist(ob); print(dist.loc, dist.scale, sep='\n')
    ac = policy.get_action(ob); print(ac)
    logp = policy.get_logp(ob); print(logp)
