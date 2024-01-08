import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
import gymnasium as gym

import wandb

from copy import copy
from icecream import ic


class Buffer():

    def __init__(self, obs_dim, act_dim, buffer_size, **kwargs):

        self.buffer_size = buffer_size
        self.device = kwargs.get('device', None)

        self.o1 = torch.zeros((buffer_size, obs_dim), device=self.device)
        self.a = torch.zeros((buffer_size, act_dim), device=self.device)
        self.r = torch.zeros((buffer_size, 1), device=self.device)
        self.o2 = torch.zeros((buffer_size, obs_dim), device=self.device)
        self.d = torch.zeros((buffer_size, 1), device=self.device)

        self.pt = 0
        self.full = False
    
    def push(self, o1, a, r, o2, d):
        
        self.o1[self.pt,:] = o1
        self.a[self.pt,:] = a
        self.r[self.pt,:] = r
        self.o2[self.pt,:] = o2
        self.d[self.pt,:] = d
        
        self.pt += 1
        if self.pt >= self.buffer_size:
            self.full = True
            self.pt = 0

    def sample(self, batch_size):

        high = self.buffer_size if self.full else self.pt
        idxs = torch.randint(0, high, (batch_size,))

        o1 = self.o1[idxs,:]
        a = self.a[idxs,:]
        r = self.r[idxs,:]
        o2 = self.o2[idxs,:]
        d = self.d[idxs,:]

        return o1, a, r, o2, d


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_size=256, activation='gelu'):

        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()

    def forward(self, x):

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x


class Agent():

    def __init__(self, obs_dim, act_dim, lr=0.001, tau=0.005, gamma=0.99, target_update=1, **kwargs):
        
        self.tau = tau
        self.gamma = gamma
        self.target_update = target_update
        self.target_update_count = 0

        self.device = kwargs.get('device')
        self.entropy_target = -act_dim

        self.p = MLP(obs_dim, act_dim * 2).to(device)
        self.p_optim = optim.Adam(self.p.parameters(), lr=lr)

        self.q1 = MLP(obs_dim + act_dim, 1).to(device)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q1t = copy(self.q1).to(device)

        self.q2 = MLP(obs_dim + act_dim, 1).to(device)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)
        self.q2t = copy(self.q2).to(device)

        self.alpha = torch.tensor([0.2], requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.alpha], lr=lr)

    def get_action(self, o, a=None):

        mean_logstd = self.p(o)
        mean = F.tanh(mean_logstd[:,:act_dim])
        logstd = torch.clip(mean_logstd[:,act_dim:], -5, 0)
        dist = torch.distributions.Normal(mean, torch.exp(logstd))

        if not a:
            a = dist.rsample()
        logp = dist.log_prob(a) - torch.log(1 - F.tanh(a).square()).sum(-1, keepdim=True)

        return a, logp
    
    def get_value(self, o, a):

        x = torch.cat((o, a), dim=-1)

        return self.q1(x), self.q2(x)
    
    def _update_target(self):

        q1_sd = self.q1.state_dict()
        q1t_sd = self.q1t.state_dict()
        for key in q1_sd.keys():
            q1t_sd[key] = (1 - self.tau) * q1t_sd[key] + self.tau * q1_sd[key]
        self.q1t.load_state_dict(q1t_sd)

        q2_sd = self.q2.state_dict()
        q2t_sd = self.q2t.state_dict()
        for key in q2_sd.keys():
            q2t_sd[key] = (1 - self.tau) * q2t_sd[key] + self.tau * q2_sd[key]
        self.q2t.load_state_dict(q2t_sd)
    
    def update(self, o1, a, r, o2, d):

        alpha = self.alpha
        a2, logp2 = self.get_action(o2)
        qt = r + self.gamma * (1 - d) * (torch.min(*self.get_value(o2, a2)) - alpha * logp2)
        q1, q2 = self.get_value(o1, a)
        q1_loss = (q1 - qt.detach()).square().mean()
        q2_loss = (q2 - qt.detach()).square().mean()

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()
        
        self.target_update_count += 1
        if self.target_update_count == self.target_update:
            self._update_target()
            self.target_update_count = 0

        a1, logp1 = self.get_action(o1)
        p_loss = (alpha.detach() * logp1 - torch.min(*self.get_value(o1, a1))).mean()

        self.p_optim.zero_grad()
        p_loss.backward()
        self.p_optim.step()

        alpha_loss = (alpha * (- logp1.detach() - self.entropy_target)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        return ((q1_loss + q2_loss) / 2).item(), p_loss.item(), alpha_loss.item(), alpha.item()


def main(args):

    wandb.init()

    device = torch.device('cuda')

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buffer = Buffer(obs_dim, act_dim, 1000000, device=device)
    agent = Agent(obs_dim, act_dim, lr=0.0003, device=device)

    gradient_steps = 1
    total_gradient_steps = 0
    while total_gradient_steps <= 1000000:

        o, _ = env.reset()
        episode_return = 0
        while True:

            with torch.no_grad():
                a, _ = agent.get_action(torch.as_tensor(o, device=device).unsqueeze(0))
            o2, r, terminated, truncated, _ = env.step(a.ravel().cpu().numpy())
            d = float(terminated or truncated)

            buffer.push(
                torch.as_tensor(o, device=device),
                torch.as_tensor(a, device=device),
                torch.as_tensor(r, device=device),
                torch.as_tensor(o2, device=device),
                torch.as_tensor(d, device=device),
            )

            o = o2
            episode_return += r
            if d:
                wandb.log({'EpisodeReturn': episode_return}, step=total_gradient_steps)

                o, _ = env.reset()
                episode_return = 0
                break

        for _ in range(gradient_steps):
            q_loss, p_loss, alpha_loss, alpha = agent.update(*buffer.sample(256))
            wandb.log({
                'Loss/Q': q_loss,
                'Loss/P': p_loss,
                'Loss/Alpha': alpha_loss,
                'Alpha': alpha,
            }, step=total_gradient_steps)
            total_gradient_steps += 1


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('env-name', type=str)

    args = parser.parse_args()

    main(args)

    pass