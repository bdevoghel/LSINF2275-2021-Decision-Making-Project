from matplotlib import pyplot as plt
import numpy as np
import re

episodes = []
mean_rewards = []
with open('logs.txt', 'r') as logs:
    for line in logs:
        if line.startswith("   - episode"):
            line = re.split(' |=', line)
            episodes.append(int(line[5]))
            mean_rewards.append(float(line[12]))

plt.scatter(episodes, mean_rewards)
plt.show()
