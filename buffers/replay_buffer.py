import numpy as np
import torch


class ReplayBuffer():

    def __init__(self, size, ob_dim, ac_dim, device=None):

        self.size = size
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim

        self.device = device

        self.idx = 0
        self.full = False

        self.ob_buf = np.zeros((size, ob_dim))
        self.ac_buf = np.zeros((size, ac_dim))
        self.rwd_buf = np.zeros(size)
        self.next_ob_buf = np.zeros((size, ob_dim))
        self.done_buf = np.zeros(size)

    def push(self, ob, ac, rwd, next_ob, done):

        len_data = 1 if ob.ndim == 1 else len(ob)
        
        if self.idx + len_data <= self.size:

            self.ob_buf[self.idx:self.idx+len_data] = ob
            self.ac_buf[self.idx:self.idx+len_data] = ac
            self.rwd_buf[self.idx:self.idx+len_data] = rwd
            self.next_ob_buf[self.idx:self.idx+len_data] = next_ob
            self.done_buf[self.idx:self.idx+len_data] = done

            self.idx += len_data

            if self.idx == self.size:
                self.full = True
                self.idx = 0

        else:

            of = self.idx + len_data - self.size  # overflow count
            self.push(ob[:-of], ac[:-of], rwd[:-of], next_ob[:-of], done[:-of])
            self.push(ob[-of:], ac[-of:], rwd[-of:], next_ob[-of:], done[-of:])

    def sample(self, batch_size):

        if self.full:
            rand_idx = np.random.choice(self.size, batch_size, replace=False)
        else:
            rand_idx = np.random.choice(self.idx, batch_size, replace=False)

        batch_ob = torch.tensor(self.ob_buf[rand_idx], dtype=torch.float32, device=self.device)
        batch_ac = torch.tensor(self.ac_buf[rand_idx], dtype=torch.float32, device=self.device)
        batch_rwd = torch.tensor(self.rwd_buf[rand_idx], dtype=torch.float32, device=self.device)
        batch_next_ob = torch.tensor(self.next_ob_buf[rand_idx], dtype=torch.float32, device=self.device)
        batch_done = torch.tensor(self.done_buf[rand_idx], dtype=torch.float32, device=self.device)

        batch = {
            'ob': batch_ob,
            'ac': batch_ac,
            'rwd': batch_rwd,
            'next_ob': batch_next_ob,
            'done': batch_done
        }

        return batch

if __name__ == '__main__':

    size = 10
    ob_dim = 3
    ac_dim = 2
    memory = ReplayBuffer(size, ob_dim, ac_dim)

    batch_size = 3
    ob = torch.randn((batch_size, ob_dim))
    ac = torch.randn((batch_size, ac_dim))
    rwd = torch.randn(batch_size)
    next_ob = torch.randn((batch_size, ob_dim))
    done = torch.randn(batch_size)

    for _ in range(9):
        memory.push(ob, ac, rwd, next_ob, done); print(memory.idx)

    batch = memory.sample(2)
