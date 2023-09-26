import os
from itertools import count

import torch
import gymnasium as gym

from agents import SACAgent


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_episodes = 1
dirname = 'trained_agents/230926_131745'
fname = 'SACAgent_80000'
agent = torch.load(os.path.join(dirname, f'{fname}.pt'), map_location=device)

envname = 'BipedalWalker-v3'
render_mode = 'human'

env = gym.make(envname, render_mode=render_mode)
rets = []
for _ in range(num_episodes):

    ret = 0
    ob, _ = env.reset()

    for _ in count():

        with torch.no_grad():

            ob_t = torch.as_tensor(ob, dtype=torch.float32, device=device).unsqueeze(0)
            ac_t, _ = agent.p.get_action(ob_t, eval=True)
            ac = ac_t.cpu().numpy().ravel()

        ob, rwd, truncated, terminated, _ = env.step(ac)
        done = truncated or terminated
        ret += rwd

        if done:
            rets.append(ret)
            break
