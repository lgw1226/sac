from copy import copy

import torch

import models


class SACAgent():

    def __init__(self, ob_dim, ac_dim, device):

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.device = device

        self.policy = models.GaussianPolicy(self.ob_dim, self.ac_dim, self.device)

        self.v = models.StateValue(self.ob_dim, self.ac_dim, self.device)
        self.v_target = copy(self.v)

        self.q1 = models.StateActionValue(self.ob_dim, self.ac_dim, self.device)
        self.q2 = models.StateActionValue(self.ob_dim, self.ac_dim, self.device)


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ob_dim = 3
    ac_dim = 2
    agent = SACAgent()
