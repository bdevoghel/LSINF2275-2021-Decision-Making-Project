import itertools
import numpy as np

import gym  # doc available here : https://gym.openai.com/docs/

from sklearn.neural_network import MLPRegressor


env = gym.make('MountainCarContinuous-v0')
"""
From https://gym.openai.com/envs/MountainCarContinuous-v0/ : A car is on a one-dimensional track, positioned 
between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong 
enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build 
up momentum. Here, the reward is greater if you spend less energy to reach the goal 
"""


class Q_learning:
    def __init__(self, n_observations=10, n_actions=20, observation_range={'speed': (-1, 1), 'position': (-1, 1)},
                 action_range=(-1, 1)):
        self.Q = np.random.rand(n_observations ** 2, n_actions)

        pos_step = (observation_range['position'][1] - observation_range['position'][0]) / n_observations
        positions = np.arange(*map(lambda x: x + pos_step, observation_range['position']), pos_step)

        speed_step = (observation_range['speed'][1] - observation_range['speed'][0]) / n_observations
        speeds = np.arange(*map(lambda x: x + speed_step, observation_range['speed']), speed_step)

        self.observations = np.array(list(itertools.product(positions, speeds)))

        action_step = (action_range[1] - action_range[0]) / n_actions
        self.actions = np.arange(*map(lambda x: x + action_step / 2, action_range), action_step)

        self.learning_rate = 0.1
        self.discout_factor = 0.95
        self.epsilon = 0.7  # exploration-exploitation factor; the greater, the more probable it is to explore (i.e. prefer random over optimal)
        self.start_epsilon = self.epsilon

        self.epsilon_decay_start = None
        self.epsilon_decay_end = None

    def observation2idx(self, observation):  # TODO to optimize (is slow)
        diff = self.observations - np.array(observation)
        norm = np.linalg.norm(diff, axis=-1)
        norm[(diff < 0).any(axis=-1)] = np.inf
        return np.argmin(norm)

    def action2value(self, action):
        return [self.actions[action]]

    def update(self, prev_observation, new_observation, action, reward):
        prev_obs = self.observation2idx(prev_observation)
        new_obs = self.observation2idx(new_observation)
        self.Q[prev_obs, action] += \
            self.learning_rate * (reward
                                  + self.discout_factor * np.max(self.Q[new_obs])
                                  - self.Q[prev_obs, action])

    def get_best_action(self, observation):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.Q[self.observation2idx(observation)])
        else:
            return np.random.randint(0, len(self.actions))

    def set_decay_values(self, epsilon_decay_start, epsilon_decay_end):
        self.epsilon_decay_start = epsilon_decay_start
        self.epsilon_decay_end = epsilon_decay_end

    def decay(self, i_episode):
        if self.epsilon_decay_start <= i_episode <= self.epsilon_decay_end:
            self.epsilon -= self.start_epsilon / (self.epsilon_decay_end - self.epsilon_decay_start)


class DeepQ_learning:
    def __init__(self, mlp_args, n_actions=20, action_range=(-1, 1)):
        self.mlp = MLPRegressor(**mlp_args)
        self.discout_factor = 0.95

        action_step = (action_range[1] - action_range[0]) / n_actions
        self.actions = np.arange(*map(lambda x: x + action_step / 2, action_range), action_step)


def q_learning(n_episodes=10000, verbose=1000):
    env.reset()

    print(
        f"Limits of observation space (position, speed)                   : low={env.observation_space.low}, high={env.observation_space.high}")
    print(
        f"Limits of action space (~acceleration)                          : low={env.action_space.low}, high={env.action_space.high}")
    print(f"Reward range (reward is inversely proportional to spent energy) : {env.reward_range}")

    agent = Q_learning(10, 20,
                       observation_range={'position': (env.observation_space.low[0], env.observation_space.high[0]),
                                          'speed': (env.observation_space.low[1], env.observation_space.high[1])},
                       action_range=(env.action_space.low, env.action_space.high))

    agent.set_decay_values(epsilon_decay_start=0, epsilon_decay_end=n_episodes // 2)
    # TODO instead of fixed nb episodes : define stopping criterion based on number of stable final states
    for i_episode in range(n_episodes):
        if i_episode % verbose == 0:
            print(f"EPISODE {i_episode + 1}/{n_episodes} - epsilon={agent.epsilon:.3f}")

        observation = env.reset()
        episode_rewards = []
        done = False
        t = 0

        while not done:
            t += 1
            if i_episode % verbose == 0:
                env.render()

            action = agent.get_best_action(observation)

            prev_observation = observation
            observation, reward, done, info = env.step(agent.action2value(action))

            agent.update(prev_observation, observation, action, reward)
            episode_rewards.append(reward)

            if done:
                if 'TimeLimit.truncated' not in info:
                    print(f"   - episode {i_episode + 1} finished after {t} timesteps with mean(episode_rewards)={np.mean(episode_rewards)}")
                break

        agent.decay(i_episode)

    env.close()


def deep_rl(n_episodes=10000, verbose=100):
    env.reset()

    mlp = MLPRegressor(hidden_layer_sizes=(5, 5), activation='logistic', solver='lbfgs', alpha=0.0001, max_iter=200, random_state=None, tol=0.0001, verbose=False, warm_start=True)

    # TODO instead of fixed nb episodes : define stopping criterion based on number of stable final states
    for i_episode in range(n_episodes):
        if i_episode % verbose == 0:
            print(f"EPISODE {i_episode + 1}/{n_episodes}")

        observation = env.reset()
        done = False
        t = 0

    #     while not done:
    #         t += 1
    #         if i_episode % verbose == 0:
    #             env.render()
    #
    #         new_q = []
    #         for action, q_sa in zip(action_space, mlp.predict(observation)):
    #             obs, r, done, info = env.step(action, apply=False)
    #             new_q.append(r + gamma * np.max(mlp.predict(obs)) - q_sa)
    #
    #         mlp.fit(np.array(observation), new_q)
    #
    #         action = action_space[np.argmax(new_q)]
    #         observation, reward, done, info = env.step(action)
    #
    #         if done:
    #             if 'TimeLimit.truncated' not in info:
    #                 print(f"   - episode {i_episode + 1} finished after {t} timesteps with reward={reward:>.6f}")
    #             break
    #
    # env.close()


if __name__ == '__main__':
    q_learning()
    # deep_rl()
