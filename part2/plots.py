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

every_episode_step = 100
every_episodes = []
mean_rewards = []
min_rewards = []
max_rewards = []
for i in range(len(rewards_with_failed) // every_episode_step):
    every_episodes.append((i + 1) * every_episode_step)
    r = rewards_with_failed[i * every_episode_step:(i + 1) * every_episode_step]
    mean_rewards.append(np.mean(r))
    min_rewards.append(np.min(r))
    max_rewards.append(np.max(r))


plt.scatter(episodes, rewards, s=5, label=f"reward of episode that succeeded")
plt.plot(every_episodes, mean_rewards, '-r', label=f"mean of rewards of last {every_episode_step} episodes")
# plt.plot(every_episodes, min_rewards, ':k', alpha=0.9, label=f"min of rewards of last {every_episode_step} episodes")
# plt.plot(every_episodes, max_rewards, ':k', alpha=0.9, label=f"max of rewards of last {every_episode_step} episodes")

plt.xlabel("episode")
plt.ylabel("reward")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1), bbox_transform=plt.gcf().transFigure)
plt.show()
