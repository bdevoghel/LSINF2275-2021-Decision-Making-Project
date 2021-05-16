from matplotlib import pyplot as plt
import numpy as np
import re
from shutil import copyfile


def load_data(filename="logs/tmp_logs.txt", nb_episodes=10000, do_record=False):
    is_tmp = filename == "logs/tmp_logs.txt"
    episodes = []
    rewards = []
    rewards_with_failed = []

    last_episode = 1
    with open(filename, 'r') as logs:
        for line in logs:
            if do_record:
                model_def = line.strip()
                filename_model_def = re.sub(':', '_', re.sub(' ', '', model_def))
                print(model_def)
                if is_tmp:
                    copyfile(filename, f"logs/{filename_model_def}.txt")
                do_record = False
            if line.startswith("AGENT"):
                do_record = True
            if line.startswith("   - episode"):
                line = re.split(" |=", line)
                episode = int(line[5])
                reward = float(line[12])

                episodes.append(episode)
                rewards.append(reward)
                rewards_with_failed += [0.0 for _ in range(episode-last_episode)]  # TODO use real reward (should be negative)
                rewards_with_failed.append(reward)
                last_episode = episode + 1
        if episode != nb_episodes:
            rewards_with_failed += [0.0 for _ in range(nb_episodes+1 - last_episode)]

    return episodes, rewards, rewards_with_failed, filename_model_def


def plot_data(episodes, rewards, rewards_with_failed: list, filename_model_def, every_episode_step=100, scatter_data=True, names=None):
    assert scatter_data == (names is None), f"{scatter_data} == {names}"

    # plt.figure(figsize=(8, 4))
    plt.figure(figsize=(6, 4))


    every_episodes = []
    mean_rewards = []
    min_rewards = []
    max_rewards = []

    for rwf in rewards_with_failed:
        mean_rewards.append([])  # for rolling avg : np.convolve(rwf, np.ones(every_episode_step), 'valid') / every_episode_step)
        min_rewards.append([])
        max_rewards.append([])
        for i in range(len(rwf) // every_episode_step):
            if len(every_episodes) < every_episode_step:
                every_episodes.append((i + 1) * every_episode_step)
            r = rwf[i * every_episode_step:(i + 1) * every_episode_step]
            mean_rewards[-1].append(np.mean(r))
            min_rewards[-1].append(np.min(r))
            max_rewards[-1].append(np.max(r))

    # for rolling average
    # every_episodes = np.arange(len(mean_rewards[-1])) + every_episode_step

    if scatter_data:
        plt.scatter(episodes, rewards, s=5, label=f"reward of episode that succeeded")
        labels = [f"average of last {every_episode_step} episodes' rewards"]
    else:
        labels = [f"{name} average" for name in names]
    colors = ['r', 'b', 'g', 'darkorange', 'm', 'y']

    for i in range(len(labels)):
        plt.plot(every_episodes, mean_rewards[i], color=colors[i], label=labels[i])
        # plt.plot(every_episodes, min_rewards[i], ':k', alpha=0.7)
        # plt.plot(every_episodes, max_rewards[i], ':k', alpha=0.7)

    print(f"\nAgent : {filename_model_def}")
    print(f"      Mean reward on the last 100 steps : {mean_rewards[-1][-1]}\n")
    
    plt.ylim((-10, 102))
    plt.xlabel("episode")
    plt.ylabel("reward")
    if names is None:
        plt.legend(loc="lower right")
    else:
        plt.legend(loc="lower right") #, bbox_to_anchor=(0.68, 0.4), bbox_transform=plt.gcf().transFigure)
    plt.savefig(f"logs/{filename_model_def[:120] if names is None else 'cmp_plot'}.png", bbox_inches="tight")
    plt.show()


compare = False
if compare:
    filenames = ["logs/results_taeq_learning.txt", 
                 "logs/results_sarsa_classic.txt", 
                 "logs/results_backward_taeq.txt", 
                 "logs/results_backward_sarsa.txt"]
    names =     ["TAE-Q-Learning's", 
                 "SARSA's", 
                 "Backward TAE-Q-Learning's",
                 "Backward SARSA's"]

    episodes = -1
    rewards = -1
    rewards_with_failed = []
    filename_model_def = []
    for i, filename in enumerate(filenames):
        episodes, rewards, rwf, fname = load_data(filename)
        rewards_with_failed.append(rwf)
        filename_model_def.append(fname)

    plot_data(episodes, None, rewards_with_failed, filename_model_def, scatter_data=False, names=names)
else:
    for filename in ["logs/results_taeq_learning.txt", 
                     "logs/results_sarsa_classic.txt", 
                     "logs/results_backward_taeq.txt", 
                     "logs/results_backward_sarsa.txt"] :
        episodes, rewards, rewards_with_failed, filename_model_def = load_data(filename)
        plot_data(episodes, rewards, [rewards_with_failed], filename_model_def)
