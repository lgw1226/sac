from copy import copy

import torch

import models


class SACAgent():

    def __init__(
        self, ob_dim, ac_dim, ac_lim,
        num_layers=2,
        layer_size=256,
        lr=0.0003,
        tau=0.005,
        gamma=0.99,
        alpha=0.1,
        device=None
    ):

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.ac_lim = ac_lim

        self.num_layers = num_layers
        self.layer_size = layer_size
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.device = device

        self.p = models.SquashedGaussianPolicy(self.ob_dim, self.ac_dim, self.ac_lim, num_layers=self.num_layers, layer_size=self.layer_size, lr=self.lr, device=self.device)

        self.q1 = models.StateActionValue(self.ob_dim, self.ac_dim, num_layers=self.num_layers, layer_size=self.layer_size, lr=self.lr, device=self.device)
        self.q2 = models.StateActionValue(self.ob_dim, self.ac_dim, num_layers=self.num_layers, layer_size=self.layer_size, lr=self.lr, device=self.device)

        self.q1_target = copy(self.q1)
        self.q2_target = copy(self.q2)

    def update_q(self, ob, ac, rwd, next_ob, done):

        q1_value = self.q1.get_value(ob, ac)
        q2_value = self.q2.get_value(ob, ac)
        with torch.no_grad():
            next_ac, next_logp = self.p.get_action(next_ob)
            q_value = torch.minimum(self.q1_target.get_value(next_ob, next_ac), self.q2_target.get_value(next_ob, next_ac))
            target = rwd + self.gamma * (1 - done) * (q_value - self.alpha * next_logp)

        q1_loss = torch.mean((q1_value - target) ** 2)
        q2_loss = torch.mean((q2_value - target) ** 2)
        self.q1.optim.zero_grad()
        self.q2.optim.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        self.q1.optim.step()
        self.q2.optim.step()

        return q1_loss + q2_loss / 2

    def update_q_target(self):

        sd1 = self.q1.net.state_dict()
        sdt1 = self.q1_target.net.state_dict()
        for key in sd1.keys():
            sdt1[key] = self.tau * sd1[key] + (1 - self.tau) * sdt1[key]

        sd2 = self.q2.net.state_dict()
        sdt2 = self.q2_target.net.state_dict()
        for key in sd2.keys():
            sdt2[key] = self.tau * sd2[key] + (1 - self.tau) * sdt2[key]

        self.q1_target.net.load_state_dict(sdt1)
        self.q2_target.net.load_state_dict(sdt2)
    
    def update_p(self, ob):

        ac, logp = self.p.get_action(ob)
        q_value = torch.minimum(self.q1_target.get_value(ob, ac), self.q2_target.get_value(ob, ac))
        
        p_loss = -torch.mean(q_value - self.alpha * logp)
        self.p.optim.zero_grad()
        p_loss.backward()
        self.p.optim.step()

        return p_loss


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ob_dim = 3
    ac_dim = 2
    agent = SACAgent(ob_dim, ac_dim, device)
