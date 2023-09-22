from math import log

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import torch.nn.functional as F


class GaussianPolicy():

    def __init__(self, ob_dim, ac_dim, ac_lim, lr=0.001, device=None):
        
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.ac_lim = ac_lim

        self.lr = lr
        self.device = device
        
        self.net = GaussianPolicyNet(self.ob_dim, self.ac_dim).to(self.device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)

    def get_dist(self, ob):

        output = self.net(ob)  # (batch_size, 2)

        mean = output[:,:self.ac_dim]
        log_std = torch.clip(output[:,self.ac_dim:], -10, 2)

        dist = distributions.Normal(mean, torch.exp(log_std))

        return dist
    
    def get_action(self, ob, eval=False):

        if not eval:
            dist = self.get_dist(ob)
            action = dist.rsample()
        else:
            dist = self.get_dist(ob)
            action = dist.loc

        return torch.tanh(action) * self.ac_lim
    
    def get_logp(self, ob):
        
        dist = self.get_dist(ob)
        ac = dist.rsample()
        logp = dist.log_prob(ac)
        logp -= 2 * (log(2) - ac - F.softplus(-2 * ac))
        
        return logp.sum(-1, keepdim=True)


class GaussianPolicyNet(nn.Module):

    def __init__(self, in_dim, out_dim):

        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim * 2)
        )

    def forward(self, x):

        return self.fc(x)
    

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ob_dim = 3
    ac_dim = 1
    policy = GaussianPolicy(ob_dim, ac_dim, device=device)

    batch_size = 4
    ob = torch.randn((batch_size, ob_dim), device=device); print(ob)
    ac_sample = policy.get_action(ob); print(ac_sample)
    ac_rsample = policy.get_reparam_action(ob); print(ac_rsample)

    # dist = policy.get_dist(ob); print(dist.loc, dist.scale, sep='\n')
    # ac = policy.get_action(ob); print(ac)
    # logp = policy.get_logp(ob, ac); print(logp)

