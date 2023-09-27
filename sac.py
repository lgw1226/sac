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

    gpu_index = args.gpu_index

    env_name = args.env_name
    num_envs = args.num_envs
    step_per_update = args.step_per_update

    memory_size = args.memory_size
    batch_size = args.batch_size
    start_step = args.start_step

    num_updates = args.num_updates
    save_count = args.save_count
    test_count = args.test_count
    num_test_episodes = args.num_test_episodes

    num_layers = args.num_layers
    layer_size = args.layer_size

    lr = args.lr
    tau = args.tau
    gamma = args.gamma
    alpha = args.alpha

    # setups

    timestamp = time.strftime('%y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{timestamp}/')

    env = gym.vector.make(env_name, num_envs=num_envs)
    ob_dim = env.single_observation_space.shape[-1]
    ac_dim = env.single_action_space.shape[-1]
    ac_lim = float(env.single_action_space.high[-1])

    device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    
    agent = agents.SACAgent(
        ob_dim, ac_dim, ac_lim,
        num_layers=num_layers,
        layer_size=layer_size,
        lr=lr,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        device=device
    )

    memory = buffers.ReplayBuffer(
        memory_size, ob_dim, ac_dim,
        device=device
    )

    # collect initial data

    ob, _ = env.reset()
    for _ in range(start_step):
        with torch.no_grad():
            ob_t = torch.as_tensor(ob, dtype=torch.float32, device=device)
            ac_t, _ = agent.p.get_action(ob_t)
            ac = ac_t.cpu().numpy()
        next_ob, rwd, terminated, truncated, _ = env.step(ac)
        done = terminated | truncated
        memory.push(ob, ac, rwd, next_ob, done)
        ob = next_ob

    # train agent

    update_count = 0

    ob, _ = env.reset()
    for _ in count():

        for _ in range(step_per_update):
            with torch.no_grad():
                ob_t = torch.as_tensor(ob, dtype=torch.float32, device=device)
                ac_t, _ = agent.p.get_action(ob_t)
                ac = ac_t.cpu().numpy()
            next_ob, rwd, terminated, truncated, _ = env.step(ac)
            done = terminated | truncated
            memory.push(ob, ac, rwd, next_ob, done)
            ob = next_ob
        
        batch = memory.sample(batch_size)
        q_loss = agent.update_q(batch['ob'], batch['ac'], batch['rwd'], batch['next_ob'], batch['done'])
        p_loss = agent.update_p(batch['ob'])
        agent.update_q_target()
        update_count += 1

        writer.add_scalar('loss/q', q_loss, update_count)
        writer.add_scalar('loss/p', p_loss, update_count)

        if update_count % test_count == 0:

            mean_return = test(env_name, agent, num_test_episodes=num_test_episodes)
            writer.add_scalar('train/mean_return', mean_return, update_count)

        if update_count % save_count == 0:

            dirname = f'trained_agents/{timestamp}/'
            os.makedirs(dirname, exist_ok=True)
            torch.save(agent, os.path.join(dirname, f'SACAgent_{update_count}.pt'))

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

            next_ob, rwd, terminated, truncated, _ = env.step(ac)
            done = terminated or truncated
            ob = next_ob
            ret += rwd

            if done:
                rets.append(ret)
                break

    return np.mean(rets)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    # cuda
    parser.add_argument('--gpu_index', type=int, default=0)

    # environment
    parser.add_argument('--env_name', type=str, default='BipedalWalker-v3')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--step_per_update', type=int, default=50)

    # agent
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--layer_size', type=int, default=256)

    # buffer
    parser.add_argument('--memory_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--start_step', type=int, default=10000)

    # training
    parser.add_argument('--num_updates', type=int, default=100000)
    parser.add_argument('--save_count', type=int, default=10000)
    parser.add_argument('--test_count', type=int, default=10)
    parser.add_argument('--num_test_episodes', type=int, default=1)

    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.2)

    args = parser.parse_args()

    train(args)