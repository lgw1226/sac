import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions


class GaussianPolicy():

    def __init__(self, ob_dim, ac_dim, device):
        
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.device = device
        
        self.net = GaussianPolicyNet(self.ob_dim).to(self.device)
        self.optim = optim.Adam(self.net.parameters())

    def get_dist(self, ob):

        output = self.net(ob)  # (batch_size, 2)

        mean = output[:,0]
        log_std = output[:,1]

        dist = distributions.Normal(mean, torch.exp(log_std))

        return dist
    
    def get_action(self, ob):

        dist = self.get_dist(ob)
        action = dist.sample()

        return action
    
    def get_logp(self, ob, ac):
        
        dist = self.get_dist(ob)
        logp = dist.log_prob(ac)

        return logp


class GaussianPolicyNet(nn.Module):

    def __init__(self, in_dim):

        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):

        return self.fc(x)
    

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ob_dim = 3
    ac_dim = 2
    policy = GaussianPolicy(ob_dim, ac_dim, device)

    batch_size = 4
    ob = torch.randn((batch_size, ob_dim), device=device); print(ob)
    dist = policy.get_dist(ob); print(dist.loc, dist.scale, sep='\n')
    ac = policy.get_action(ob); print(ac)
    logp = policy.get_logp(ob, ac); print(logp)
