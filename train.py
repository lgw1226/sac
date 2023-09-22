import os
import time
from itertools import count

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import agents
import buffers


def train():

    env_name = 'Pendulum-v1'
    memory_size = 100000
    batch_size = 256
    num_update = 100000
    save_count = 10000
    test_count = 1000

    timestamp = time.strftime('%y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{timestamp}/')

    render_mode = None
    env = gym.make(env_name, render_mode=render_mode)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    agent = agents.SACAgent(ob_dim, ac_dim, device=device)
    memory = buffers.ReplayBuffer(memory_size, ob_dim, ac_dim, device=device)

    while not memory.full:

        ob, _ = env.reset()

        for _ in count():
            with torch.no_grad():
                ob_t = torch.as_tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
                ac = agent.p.get_action(ob_t).view(1).cpu().numpy()
            next_ob, rwd, terminated, truncated, _ = env.step(ac)
            done = terminated or truncated
            memory.push(ob, ac, rwd, next_ob, done)
            ob = next_ob

            if done:
                break

    update_count = 0

    for _ in count():

        ob, _ = env.reset()

        for _ in count():

            with torch.no_grad():
                ob_t = torch.as_tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
                ac = agent.p.get_action(ob_t).view(1).cpu().numpy()
            next_ob, rwd, terminated, truncated, _ = env.step(ac)
            done = terminated or truncated
            memory.push(ob, ac, rwd, next_ob, done)
            ob = next_ob

            if done: break
        
        batch = memory.sample(batch_size)
        v_loss = agent.update_v(batch['ob'])
        q1_loss, q2_loss = agent.update_q(batch['ob'], batch['ac'], batch['rwd'], batch['next_ob'])
        p_loss = agent.update_p(batch['ob'])
        agent.update_v_target()
        update_count += 1

        writer.add_scalar('loss/v', v_loss, update_count)
        writer.add_scalar('loss/q1', q1_loss, update_count)
        writer.add_scalar('loss/q2', q2_loss, update_count)
        writer.add_scalar('loss/p', p_loss, update_count)

        if update_count % test_count == 0:

            mean_return = test(env, agent)
            writer.add_scalar('test/mean_return', mean_return, update_count)

        if update_count % save_count == 0:

            dirname = f'trained_agents/{timestamp}/'
            os.makedirs(dirname, exist_ok=True)
            torch.save(agent, os.path.join(dirname, f'SACAgent_{update_count}.pt'))

        if update_count == num_update: break

def test(env, agent, num_test=10):

    device = agent.device
    rets = []

    for _ in range(num_test):

        ret = 0
        ob, _ = env.reset()

        for _ in count():

            with torch.no_grad():
                ob_t = torch.as_tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
                ac = agent.p.get_action(ob_t).view(1).cpu().numpy()

            next_ob, rwd, terminated, truncated, _ = env.step(ac)
            done = terminated or truncated
            ob = next_ob
            ret += rwd

            if done:
                rets.append(ret)
                break

    return np.mean(rets)


if __name__ == '__main__':

    train()