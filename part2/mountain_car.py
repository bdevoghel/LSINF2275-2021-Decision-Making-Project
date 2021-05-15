import itertools
import numpy as np

import gym  # doc available here : https://gym.openai.com/docs/

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


env = gym.make('MountainCarContinuous-v0')
"""
From https://gym.openai.com/envs/MountainCarContinuous-v0/ : A car is on a one-dimensional track, positioned 
between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong 
enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build 
up momentum. Here, the reward is greater if you spend less energy to reach the goal 
"""


class Agent:
    def __init__(self):
        self.observation_range = None
        self.action_range = None

        self.epsilon = 0.5
        self.discount_factor = 0.95
        self.start_epsilon = self.epsilon

        self.epsilon_decay_start = None
        self.epsilon_decay_end = None

    def action2value(self, action):
        raise NotImplementedError("action2value not implemented")

    def update(self, prev_observation, action, new_observation, reward):
        raise NotImplementedError("update not implemented")

    def get_best_action(self, observation):
        raise NotImplementedError("get_best_action not implemented")

    def set_decay_values(self, epsilon_decay_start, epsilon_decay_end):
        self.epsilon_decay_start = epsilon_decay_start
        self.epsilon_decay_end = epsilon_decay_end

    def decay(self, i_episode):
        if self.epsilon_decay_start <= i_episode <= self.epsilon_decay_end:
            self.epsilon -= self.start_epsilon / (self.epsilon_decay_end - self.epsilon_decay_start)


class Q_learning(Agent):
    def __init__(self, n_observations=30, n_actions=10, observation_range={'speed': (-1, 1), 'position': (-1, 1)},
                 action_range=(-1, 1)):
        Agent.__init__(self)

        self.Q = np.random.rand(n_observations**2, n_actions)

        pos_step = (observation_range['position'][1] - observation_range['position'][0]) / n_observations
        positions = np.arange(*map(lambda x: x + pos_step, observation_range['position']), pos_step)

        speed_step = (observation_range['speed'][1] - observation_range['speed'][0]) / n_observations
        speeds = np.arange(*map(lambda x: x + speed_step, observation_range['speed']), speed_step)

        self.observations = np.array(list(itertools.product(positions, speeds)))

        action_step = (action_range[1] - action_range[0]) / n_actions
        self.actions = np.arange(*map(lambda x: x + action_step / 2, action_range), action_step)

        self.learning_rate = 3e-2

    def observation2idx(self, observation):  # TODO to optimize (is slow)
        diff = self.observations - np.array(observation)
        norm = np.linalg.norm(diff, axis=-1)
        norm[(diff < 0).any(axis=-1)] = np.inf
        return np.argmin(norm)

    def action2value(self, action):
        return [self.actions[action]]

    def update(self, prev_observation, action, new_observation, reward):
        prev_obs = self.observation2idx(prev_observation)
        new_obs = self.observation2idx(new_observation)
        self.Q[prev_obs, action] += \
            self.learning_rate * (reward
                                  + self.discount_factor * np.max(self.Q[new_obs])
                                  - self.Q[prev_obs, action])

    def get_best_action(self, observation):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.Q[self.observation2idx(observation)])
        else:
            return np.random.randint(0, len(self.actions))


class DeepQ_learning(Agent):
    def __init__(self, mlp_args, n_actions=20, action_range=(-1, 1), observation_range={'speed': (-1, 1), 'position': (-1, 1)}):
        Agent.__init__(self)
        self.observation_range = observation_range
        self.action_range = action_range
        self.mlp = MLPRegressor(**mlp_args)

        self.x, self.y = [], []
        self.batch_size = 1500

        action_step = (action_range[1] - action_range[0]) / n_actions
        self.actions = np.arange(*map(lambda x: x + action_step / 2, action_range), action_step)

        self.first_fit()

        self.scaler = None

    def first_fit(self):
        x = np.array([[np.mean(self.observation_range['position']), np.mean(self.observation_range['speed'])]])
        y = np.random.uniform(*self.action_range, size=(1, len(self.actions)))
        self.mlp.fit(x, y)

    def get_best_action(self, observation):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.mlp.predict(observation[None, :])[0, :])
        else:
            return np.random.randint(0, len(self.actions))

    def action2value(self, action):
        return [self.actions[action]]

    def update(self, prev_observation, action, new_observation, reward):
        old_Q = self.mlp.predict(prev_observation[None, :])[0, :]
        new_Q = self.mlp.predict(new_observation[None, :])[0, :]

        old_Q[action] = reward + self.discount_factor * np.max(new_Q)

        self.x.append(prev_observation)
        self.y.append(old_Q)

        if len(self.x) == self.batch_size:
            self.learn()

    def learn(self):
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(np.array(self.x))
        self.mlp.fit(self.scaler.transform(np.array(self.x)), np.array(self.y))
        self.x, self.y = [], []


def learning(agent, n_episodes=10000, verbose=1000):
    env.reset()
    print(
        f"Limits of observation space (position, speed)                   : low={env.observation_space.low}, high={env.observation_space.high}")
    print(
        f"Limits of action space (~acceleration)                          : low={env.action_space.low}, high={env.action_space.high}")
    print(f"Reward range (reward is inversely proportional to spent energy) : {env.reward_range}")

    agent.set_decay_values(epsilon_decay_start=0, epsilon_decay_end=n_episodes)

    for i_episode in range(n_episodes):
        if i_episode % verbose == 0:
            print(f"EPISODE {i_episode + 1}/{n_episodes}")
        observation = env.reset()
        done = False
        t = 0
        while not done:
            t += 1
            if i_episode % verbose == 0:
                env.render()

            action = agent.get_best_action(observation)
            prev_observation = observation

            observation, reward, done, info = env.step(agent.action2value(action))
            agent.update(prev_observation, action, observation, reward)

            if done:
                if 'TimeLimit.truncated' not in info:
                    print(f"   - episode {i_episode + 1} finished after {t + 1} timesteps with reward={reward:>.6f}")
                break
        agent.decay(i_episode)
    env.close()


if __name__ == '__main__':
    q_agent = Q_learning(30, 10,
                         observation_range={'position': (env.observation_space.low[0], env.observation_space.high[0]),
                                            'speed': (env.observation_space.low[1], env.observation_space.high[1])},
                         action_range=(env.action_space.low, env.action_space.high))

    mlp_args = {'hidden_layer_sizes': (8, 8),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'max_iter': 10000,
                'random_state': None,
                'tol': 0.0001,
                'verbose': False,
                'warm_start': True}

    deep_agent = DeepQ_learning(mlp_args, n_actions=10,
                                action_range=(env.action_space.low, env.action_space.high),
                                observation_range={'position': (env.observation_space.low[0], env.observation_space.high[0]),
                                                   'speed': (env.observation_space.low[1], env.observation_space.high[1])})

    learning(q_agent, verbose=100)
