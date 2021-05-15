from matplotlib import pyplot as plt
import numpy as np
import re

episodes = []
rewards = []
rewards_with_failed = []
last_episode = 1
with open('logs.txt', 'r') as logs:
    for line in logs:
        if line.startswith("   - episode"):
            line = re.split(' |=', line)
            episode = int(line[5])
            reward = float(line[12])

            episodes.append(episode)
            rewards.append(reward)
            rewards_with_failed += [0.0 for _ in range(episode-last_episode)]  # TODO use real reward (should be negative)
            rewards_with_failed.append(reward)
            last_episode = episode + 1

mean_every_episode = 100
mean_episodes = []
mean_rewards = []
for i in range(len(rewards_with_failed)//mean_every_episode):
    mean_episodes.append((i+1) * mean_every_episode)
    mean_rewards.append(np.mean(rewards_with_failed[i*mean_every_episode:(i+1)*mean_every_episode]))


plt.scatter(episodes, rewards)
plt.plot(mean_episodes, mean_rewards, 'r-')
plt.show()
