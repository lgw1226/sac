from itertools import count

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


render_mode = 'human'

env = gym.make('Pendulum-v1', render_mode=render_mode)

num_episode = 3
rets = []
for i in range(num_episode):
    ret = 0
    ob, info = env.reset()
    for _ in count():
        ac = env.action_space.sample()
        ob, rwd, terminated, truncated, info = env.step(ac)
        ret += rwd

        if terminated or truncated:
            rets.append(ret)
            break

rets = np.array(rets)
print(f"Mean return of {num_episode} episodes: {np.mean(rets):.2f}")

plt.figure()
plt.plot(rets, 'k-')
plt.title('Pendulum-v1')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.show()