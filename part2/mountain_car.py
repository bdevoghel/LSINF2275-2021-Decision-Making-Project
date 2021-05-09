import itertools
import numpy as np

import gym  # doc available here : https://gym.openai.com/docs/

env = gym.make('MountainCarContinuous-v0')
"""
From https://gym.openai.com/envs/MountainCarContinuous-v0/ : A car is on a one-dimensional track, positioned 
between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong 
enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build 
up momentum. Here, the reward is greater if you spend less energy to reach the goal 
"""


class Q_learning:
    def __init__(self, n_observations=10, n_actions=20, observation_range={'speed': (-1, 1), 'position': (-1, 1)}, action_range=(-1, 1)):
        self.Q = np.random.rand(n_observations**2, n_actions)

        step = (observation_range['position'][1]-observation_range['position'][0])/n_observations
        positions = np.arange(*map(lambda x: x+step, observation_range['position']), step)

        step = (observation_range['speed'][1]-observation_range['speed'][0])/n_observations
        speeds = np.arange(*map(lambda x: x+step, observation_range['speed']), step)

        self.observations = np.array(list(itertools.product(positions, speeds)))

        step = (action_range[1]-action_range[0])/n_actions
        self.actions = np.arange(*map(lambda x: x+step/2, action_range), step)

        self.learning_rate = 0.1
        self.discout_factor = 0.95
        self.epsilon = 0.5

        self.epsilon_decay_start = None
        self.epsilon_decay_end = None

    def observation2idx(self, observation):
        diff = self.observations - np.array(observation)
        norm = np.linalg.norm(diff, axis=-1)
        norm[(diff < 0).any(axis=-1)] = np.inf
        return np.argmin(norm)

    def action2value(self, action):
        return [self.actions[action]]

    def update(self, prev_observation, new_observation, action, reward):
        self.Q[self.observation2idx(prev_observation), action] += \
            self.learning_rate * (reward
                                  + self.discout_factor * np.max(self.Q[self.observation2idx(new_observation)])
                                  - self.Q[self.observation2idx(prev_observation), action])

    def get_action(self, observation):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.Q[self.observation2idx(observation)])
        else:
            return np.random.randint(0, len(self.actions))

    def set_decay_values(self, n_episodes, explore_until):
        self.epsilon_decay_start = 0
        self.epsilon_decay_end = explore_until

    def decay(self, i_episode):
        if self.epsilon_decay_end >= i_episode >= self.epsilon_decay_start:
            self.epsilon -= self.epsilon/(self.epsilon_decay_end - self.epsilon_decay_start)


if __name__ == '__main__':
    env.reset()

    print(f"Limits of observation space (position, speed)                   : low={env.observation_space.low}, high={env.observation_space.high}")
    print(f"Limits of action space (~acceleration)                          : low={env.action_space.low}, high={env.action_space.high}")
    print(f"Reward range (reward is inversely proportional to spent energy) : {env.reward_range}")

    Q = Q_learning(10, 20,
                      observation_range={'position': (env.observation_space.low[0], env.observation_space.high[0]),
                                         'speed': (env.observation_space.low[1], env.observation_space.high[1])},
                      action_range=(env.action_space.low, env.action_space.high))

    n_episodes = 10000
    Q.set_decay_values(n_episodes, explore_until=3000)
    for i_episode in range(n_episodes):
        if i_episode % 100 == 0:
            print(f"EPISODE {i_episode + 1}/{n_episodes}")
        observation = env.reset()
        # print(observation)

        done = False
        t = 0
        while not done:
            t += 1
            if i_episode % 200 == 0:
                env.render()

            prev_observation = observation
            action = Q.get_action(observation)

            observation, reward, done, info = env.step(Q.action2value(action))
            # print(observation, reward)

            Q.update(prev_observation, observation, action, reward)

            if done:
                if 'TimeLimit.truncated' not in info:
                    print(f"   - episode {i_episode+1} finished after {t+1} timesteps with reward={reward:>.6f}")
                break

        Q.decay(i_episode)

    env.close()

