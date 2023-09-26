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


def train(args):

    env_name = args.env_name

    num_updates = args.num_updates
    save_count = args.save_count
    test_count = args.test_count
    num_test_episodes = args.num_test_episodes

    num_layers = args.num_layers
    layer_size = args.layer_size

    lr = args.lr
    gamma = args.gamma

    # setups

    timestamp = time.strftime('%y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{timestamp}/')

    env = gym.make(env_name)
    ob_dim = env.observation_space.shape[-1]
    ac_dim = env.action_space.shape[-1]
    ac_lim = float(np.min(env.action_space.high))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    agent = agents.VPGAgent(
        ob_dim, ac_dim, ac_lim,
        num_layers=num_layers,
        layer_size=layer_size,
        lr=lr,
        gamma=gamma,
        device=device
    )

    # train agent

    update_count = 0

    for _ in count():

        obs = []
        rwds = []
        ob, _ = env.reset()

        for _ in count():

            with torch.no_grad():
                ob_t = torch.as_tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
                ac_t, _ = agent.p.get_action(ob_t)
                ac = ac_t.cpu().numpy().ravel()
            obs.append(ob)
            ob, rwd, terminated, truncated, _ = env.step(ac)
            done = terminated or truncated
            rwds.append(rwd)

            if done: break
        
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32, device=device)
        rtgs = agent.rtgs(np.array(rwds))
        p_loss = agent.update_p(obs_t)
        v_loss = agent.update_v(obs_t, rtgs)
        update_count += 1

        writer.add_scalar('loss/p', p_loss, update_count)
        writer.add_scalar('loss/v', v_loss, update_count)

        if update_count % test_count == 0:

            mean_return = test(env_name, agent, num_test_episodes=num_test_episodes)
            writer.add_scalar('train/mean_return', mean_return, update_count)

        if update_count % save_count == 0:

            dirname = f'trained_agents/{timestamp}/'
            os.makedirs(dirname, exist_ok=True)
            torch.save(agent, os.path.join(dirname, f'VPGAgent_{update_count}.pt'))

        if update_count == num_updates: break

def test(env_name, agent, num_test_episodes=1):

    device = agent.device
    env = gym.make(env_name)

    rets = []
    for _ in range(num_test_episodes):
        ret = 0
        ob, _ = env.reset()
        for _ in count():

            with torch.no_grad():

                ob_t = torch.as_tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
                ac_t, _ = agent.p.get_action(ob_t, eval=True)
                ac = ac_t.cpu().numpy().ravel()

            ob, rwd, terminated, truncated, _ = env.step(ac)
            done = terminated or truncated
            ret += rwd

            if done:
                rets.append(ret)
                break

    return np.mean(rets)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--env_name', type=str, default='Pendulum-v1')

    # agent
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--layer_size', type=int, default=64)

    # training
    parser.add_argument('--num_updates', type=int, default=100000)
    parser.add_argument('--save_count', type=int, default=10000)
    parser.add_argument('--test_count', type=int, default=10)
    parser.add_argument('--num_test_episodes', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)

    args = parser.parse_args()

    train(args)