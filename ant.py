from itertools import count

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


render_mode = None

env = gym.make('Ant-v4', render_mode=render_mode)

num_episode = 100
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
plt.title('Ant-v4')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('ant_random_action')
plt.show()