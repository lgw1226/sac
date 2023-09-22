from copy import copy

import torch

import models


class SACAgent():

    def __init__(
        self, ob_dim, ac_dim,
        lr=0.0003,
        tau=0.005,
        gamma=0.99,
        alpha=0.1,
        device=None
    ):

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.device = device

        self.p = models.GaussianPolicy(self.ob_dim, self.ac_dim, lr=self.lr, device=self.device)

        self.v = models.StateValue(self.ob_dim, lr=self.lr, device=self.device)
        self.v_target = copy(self.v)

        self.q1 = models.StateActionValue(self.ob_dim, self.ac_dim, lr=self.lr, device=self.device)
        self.q2 = models.StateActionValue(self.ob_dim, self.ac_dim, lr=self.lr, device=self.device)

    def update_v(self, ob):

        value = self.v.get_value(ob)
        with torch.no_grad():
            ac = self.p.get_action(ob)
            q_value = torch.minimum(self.q1.get_value(ob, ac), self.q2.get_value(ob, ac))
            logp = self.p.get_logp(ob, ac)

        v_loss = torch.mean(0.5 * (value - q_value + self.alpha * logp) ** 2)
        self.v.optim.zero_grad()
        v_loss.backward()
        self.v.optim.step()

        return v_loss

    def update_v_target(self):

        sd = self.v.net.state_dict()
        sd_target = self.v_target.net.state_dict()

        for key in sd.keys():
            sd_target[key] = self.tau * sd[key] + (1 - self.tau) * sd_target[key]

    def update_q(self, ob, ac, rwd, next_ob):

        q1_value = self.q1.get_value(ob, ac)
        q2_value = self.q2.get_value(ob, ac)
        with torch.no_grad():
            value_target = self.v_target.get_value(next_ob)

        q1_loss = torch.mean(0.5 * (q1_value - (rwd + self.gamma * value_target)) ** 2)
        q2_loss = torch.mean(0.5 * (q2_value - (rwd + self.gamma * value_target)) ** 2)
        self.q1.optim.zero_grad()
        self.q2.optim.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        self.q1.optim.step()
        self.q2.optim.step()

        return q1_loss, q2_loss
    
    def update_p(self, ob):

        with torch.no_grad():
            ac = self.p.get_action(ob)
            q_value = torch.minimum(self.q1.get_value(ob, ac), self.q2.get_value(ob, ac))
        logp = self.p.get_logp(ob, ac)
        
        p_loss = torch.mean(logp - q_value)
        self.p.optim.zero_grad()
        p_loss.backward()
        self.p.optim.step()

        return p_loss


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ob_dim = 3
    ac_dim = 2
    agent = SACAgent(ob_dim, ac_dim, device)
