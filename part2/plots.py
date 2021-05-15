from matplotlib import pyplot as plt
import numpy as np
import re

episodes = np.ones(10000)
with open('logs_w_energy_consumption.txt', 'r') as logs:
    for line in logs:
        if line.startswith("   - episode"):
            line = re.split(' |=', line)
            episodes[int(line[5]) - 1] = float(line[8])

plt.plot(np.arange(len(episodes))[-1000:], episodes[-1000:], 'o')
plt.show()
