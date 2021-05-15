import itertools
import numpy as np

import gym  # doc available here : https://gym.openai.com/docs/

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import pickle


env = gym.make('MountainCarContinuous-v0')
"""
From https://gym.openai.com/envs/MountainCarContinuous-v0/ : A car is on a one-dimensional track, positioned 
between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong 
enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build 
up momentum. Here, the reward is greater if you spend less energy to reach the goal 
"""


class Agent:
    def __init__(self, epsilon=0.5, discount_factor=0.95):
        self.name = "AbstractAgent"

        self.observation_range = None
        self.action_range = None

        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.start_epsilon = self.epsilon

        self.epsilon_decay_start = None
        self.epsilon_decay_end = None

        self.t = 1
        self.T = 1e5
        self.start_T = self.T
        self.step = 0

    def action2value(self, action):
        raise NotImplementedError("action2value not implemented")

    def update(self, prev_observation, action, new_observation, reward, done, info):
        raise NotImplementedError("update not implemented")

    def get_best_action(self, observation):
        raise NotImplementedError("get_best_action not implemented")

    def set_decay_values(self, epsilon_decay_start, epsilon_decay_end):
        self.epsilon_decay_start = epsilon_decay_start
        self.epsilon_decay_end = epsilon_decay_end

    def decay(self):
        self.t += 1
        self.T = self.start_T/np.log(self.t)
        if self.epsilon_decay_start <= self.t <= self.epsilon_decay_end:
            self.epsilon -= self.start_epsilon / (self.epsilon_decay_end - self.epsilon_decay_start)

    def new_episode(self):
        pass

    def get_parameters(self):
        return {"epsilon": self.epsilon,
                "discount_factor": self.discount_factor,
                "epsilon_decay_start": self.epsilon_decay_start,
                "epsilon_decay_end": self.epsilon_decay_end}

    def verbose_episode(self):
        raise NotImplementedError("verbose_episode not implemented (best practice : put changing parameters")

    def save(self):
        file = open(f'pickle/{self.__class__.__name__}.p', 'wb')
        pickle.dump(self, file)
        file.close()


class QLearning(Agent):
    def __init__(self, epsilon, discount_factor, learning_rate=3e-2, n_observations=30, n_actions=10,
                 observation_range={'speed': (-1, 1), 'position': (-1, 1)},
                 action_range=(-1, 1), action_strategy='boltzmann', init_strategy='random'):
        Agent.__init__(self, epsilon, discount_factor)
        self.name = self.__class__.__name__

        self.observation2idx_cache = {}

        self.init_strategy = init_strategy
        self.Q = np.random.rand(n_observations**2, n_actions) if init_strategy == 'random' else np.full((n_observations**2, n_actions), -np.inf)

        pos_step = (observation_range['position'][1] - observation_range['position'][0]) / n_observations
        positions = np.arange(*map(lambda x: x + pos_step, observation_range['position']), pos_step)

        speed_step = (observation_range['speed'][1] - observation_range['speed'][0]) / n_observations
        speeds = np.arange(*map(lambda x: x + speed_step, observation_range['speed']), speed_step)

        self.observations = np.array(list(itertools.product(positions, speeds)))

        action_step = (action_range[1] - action_range[0]) / n_actions
        self.actions = np.arange(*map(lambda x: x + action_step / 2, action_range), action_step)

        self.learning_rate = learning_rate

        self.use_boltzmann = action_strategy == 'boltzmann'

    def observation2idx(self, observation):  # TODO to optimize (is slow)
        obs_bytes = observation.tobytes()
        if obs_bytes in self.observation2idx_cache:
            return self.observation2idx_cache[obs_bytes]

        diff = self.observations - np.array(observation)
        norm = np.linalg.norm(diff, axis=-1)
        norm[(diff < 0).any(axis=-1)] = np.inf
        result = np.argmin(norm)

        self.observation2idx_cache[obs_bytes] = result
        return result

    def action2value(self, action):
        return [self.actions[action]]

    def update(self, prev_observation, action, new_observation, reward, done, info):
        prev_obs = self.observation2idx(prev_observation)
        new_obs = self.observation2idx(new_observation)
        future_reward = np.max(self.Q[new_obs])
        self.Q[prev_obs, action] += \
            self.learning_rate * (reward
                                  + self.discount_factor * future_reward
                                  - self.Q[prev_obs, action])

    def get_best_action(self, observation):
        if self.use_boltzmann:
            return np.random.choice(a=np.arange(0, len(self.actions)), p=self.boltzmann(self.observation2idx(observation)))
        else:
            if np.random.rand() > self.epsilon:
                return np.argmax(self.Q[self.observation2idx(observation)])
            else:
                return np.random.randint(0, len(self.actions))

    def boltzmann(self, s):
        e = np.exp(self.Q[s, :]/self.T)
        return e/np.sum(e)

    def get_parameters(self):
        return {**Agent.get_parameters(self),
                "learning_rate": self.learning_rate,
                "action_strategy": "boltzmann" if self.use_boltzmann else "simulated annealing",
                "init_strategy":self.init_strategy}

    def verbose_episode(self):
        return f"epsilon={self.epsilon:.4f}"


class SARSA(QLearning):
    def __init__(self, epsilon=0.5, discount_factor=0.95, learning_rate=0.03, n_observations=30, n_actions=10,
                 observation_range={'speed': (-1, 1), 'position': (-1, 1)},
                 action_range=(-1, 1)):
        QLearning.__init__(self, epsilon, discount_factor, learning_rate, n_observations, n_actions, observation_range, action_range)
        self.cached_action = None
        self.cached_obs = None

    def get_best_action(self, observation, cached=True):
        if not cached or self.cached_action is None:
            self.cached_obs = observation
            self.cached_action = QLearning.get_best_action(self, observation)
            return self.cached_action
        else:
            if not np.all(self.cached_obs == observation):
                raise ValueError('wrong observation')
            return self.cached_action

    def update(self, prev_observation, action, new_observation, reward, done, info):
        prev_obs = self.observation2idx(prev_observation)
        new_obs = self.observation2idx(new_observation)
        future_reward = self.Q[new_obs, self.get_best_action(new_observation, cached=False)]

        self.Q[prev_obs, action] += \
            self.learning_rate * (reward
                                  + self.discount_factor * future_reward
                                  - self.Q[prev_obs, action])

    def new_episode(self):
        self.cached_action = None
        self.cached_obs = None


class NStepSARSA(QLearning):
    def __init__(self, epsilon=0.5, discount_factor=0.95, learning_rate=0.03, n_observations=30, n_actions=10, observation_range={'speed': (-1, 1), 'position': (-1, 1)},
                 action_range=(-1, 1), action_strategy='simulated annealing', lookahead=5):
        QLearning.__init__(self, epsilon, discount_factor, learning_rate, n_observations, n_actions, observation_range, action_range, action_strategy)
        self.cached_actions = [-1]
        self.cached_states = []
        self.cached_rewards = [0.]
        self.M = np.inf
        self.lookahead = lookahead

    def get_best_action(self, observation):
        action = QLearning.get_best_action(self, observation)
        self.cached_actions.append(action)
        return action

    def update(self, prev_observation, action, new_observation, reward, done, info):
        # https://medium.com/zero-equals-false/n-step-td-method-157d3875b9cb
        if self.step == 0:
            self.cached_states.append(prev_observation)

        # goal reached, finishing to update Q
        self.step = self.step if done is not None else self.step + 1

        if self.step < self.M:
            # self.cached_actions.append(action)
            self.cached_rewards.append(reward)
            self.cached_states.append(new_observation)

            if done:
                self.M = self.step + 1

        tau = self.step - self.lookahead + 1

        if tau >= 0:
            G = 0.  # expected reward
            for i in range(tau+1, min(tau+self.lookahead, self.M) + 1):
                G += self.discount_factor ** (i - tau - 1) * self.cached_rewards[i]
            if tau + self.lookahead < self.M:
                G += self.discount_factor ** self.lookahead * self.Q[self.observation2idx(self.cached_states[tau+self.lookahead]), self.cached_actions[tau+self.lookahead]]
            self.Q[self.observation2idx(self.cached_states[tau]), self.cached_actions[tau]] += self.learning_rate * (G - self.Q[self.observation2idx(self.cached_states[tau]), self.cached_actions[tau]])

        if self.step + 1 >= self.M and tau < self.M - 1:
            self.update(None, None, None, None, None)  # reached end of episode but needs to update Q

    def new_episode(self):
        self.cached_actions = [-1]
        self.cached_states = []
        self.cached_rewards = [0.]
        self.M = np.inf

    def get_parameters(self):
        return {**QLearning.get_parameters(self),
                "lookahead": self.lookahead}


class DeepQLearning(Agent):
    def __init__(self, mlp_args, epsilon, discount_factor, batch_size=1500, n_actions=20,
                 observation_range={'speed': (-1, 1), 'position': (-1, 1)},
                 action_range=(-1, 1)):
        Agent.__init__(self, epsilon, discount_factor)
        self.name = self.__class__.__name__

        self.observation_range = observation_range
        self.action_range = action_range
        self.mlp = MLPRegressor(**mlp_args)

        self.x, self.y = [], []
        self.batch_size = batch_size
        self.mlp_args = mlp_args

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

    def update(self, prev_observation, action, new_observation, reward, done, info):
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

    def get_parameters(self):
        return {**Agent.get_parameters(self),
                "mlp_args": mlp_args,
                "batch_size": self.batch_size}

    def verbose_episode(self):
        return f"epsilon={self.epsilon:.4f}"


class BackwardsSARSA(QLearning):
    def __init__(self, epsilon=0.5, discount_factor=0.95, learning_rate=0.03, n_observations=30, n_actions=10, backwards_learning_rate=0.1, backwards_discount_factor=0.9,
                 observation_range={'speed': (-1, 1), 'position': (-1, 1)},
                 action_range=(-1, 1)):
        QLearning.__init__(self, epsilon, discount_factor, learning_rate, n_observations, n_actions, observation_range, action_range)

        self.backwards_learning_rate = backwards_learning_rate
        self.backwards_discount_factor = backwards_discount_factor
        self.M = []

    def update(self, prev_observation, action, new_observation, reward, done, info):
        prev_obs_idx = self.observation2idx(prev_observation)
        new_obs_idx = self.observation2idx(new_observation)
        future_reward = self.Q[new_obs_idx, self.get_best_action(new_observation)]

        # store values for i
        self.M[-1].append((prev_obs_idx, action, reward, new_obs_idx))

        if done and 'TimeLimit.truncated' not in info:
            # terminal state (eq 11)
            for j in range(len(self.M)):
                for t in range(len(self.M[-j])):
                    prev_obs_idx, action, reward, new_obs_idx = self.M[-j][t]
                    new_observation = self.observation2idx(new_obs_idx)
                    future_reward = self.Q[new_obs_idx, self.get_best_action(new_observation)]
                    self.Q[prev_obs_idx, action] += \
                        self.backwards_learning_rate * (reward
                                                        + self.backwards_discount_factor * future_reward
                                                        - self.Q[prev_obs_idx, action])

        else:
            # non-terminal state (eq 9)
            self.Q[prev_obs_idx, action] += \
                self.learning_rate * (reward
                                      + self.discount_factor * future_reward
                                      - self.Q[prev_obs_idx, action])

    def new_episode(self):
        self.M.append([])

    def get_parameters(self):
        return {**QLearning.get_parameters(self),
                  "backwards_learning_rate": self.backwards_learning_rate,
                  "backwards_discount_factor": self.backwards_discount_factor}


def learning(agent:Agent, n_episodes:int, verbose=1000):
    env.reset()
    print("ENVIRONMENT : ")
    print(f"   Limits of observation space (position, speed)                   : " +
                f"low={env.observation_space.low}, high={env.observation_space.high}")
    print(f"   Limits of action space (~acceleration)                          : " +
                f"low={env.action_space.low}, high={env.action_space.high}")
    print(f"   Reward range (reward is inversely proportional to spent energy) : " +
                f"{env.reward_range}")

    agent.set_decay_values(epsilon_decay_start=0, epsilon_decay_end=n_episodes)
    print(f"AGENT : \n   {agent.name} : {agent.get_parameters()}")

    # repeat for each episode
    for i_episode in range(n_episodes):
        if i_episode % verbose == 0:
            print(f"EPISODE {i_episode + 1}/{n_episodes} - {agent.verbose_episode()}")
        observation = env.reset()
        agent.new_episode()
        done = False
        episode_rewards = []
        agent.step = 0
        while not done:
            if i_episode % verbose == 0:
                env.render()

            # choose action
            action = agent.get_best_action(observation)
            prev_observation = observation

            # execute action
            observation, reward, done, info = env.step(agent.action2value(action))
            observation = np.round(observation, 5)
            episode_rewards.append(reward)

            # learn
            agent.update(prev_observation, action, observation, reward, done, info)

            if done:
                if 'TimeLimit.truncated' not in info:
                    print(f"   - episode {i_episode + 1} finished after {agent.step} timesteps with sum(episode_rewards)={np.sum(episode_rewards)}")
                break
            agent.step += 1
        agent.decay()
    env.close()


if __name__ == '__main__':
    observation_range = {'position': (env.observation_space.low[0], env.observation_space.high[0]),
                         'speed': (env.observation_space.low[1], env.observation_space.high[1])}
    action_range = (env.action_space.low, env.action_space.high)

    q_agent = QLearning(epsilon=0.5, discount_factor=0.99, learning_rate=0.01, n_observations=30, n_actions=10,
                         observation_range=observation_range,
                         action_range=action_range,
                         action_strategy='simulated annealing')

    sarsa_agent = SARSA(epsilon=0.5, discount_factor=0.99, learning_rate=0.02,
                        n_observations=30, n_actions=10,
                        observation_range=observation_range,
                        action_range=action_range)

    nstep_sarsa_agent = NStepSARSA(lookahead=10, epsilon=0.5, discount_factor=0.99, learning_rate=0.02,
                                   n_observations=30, n_actions=10,
                                   observation_range=observation_range,
                                   action_range=action_range)

    backwards_sarsa_agent = BackwardsSARSA(epsilon=0.5, discount_factor=0.9999, learning_rate=0.015, n_observations=30,
                                           n_actions=10, backwards_learning_rate=0.01, backwards_discount_factor=0.9999,
                                           observation_range=observation_range,
                                           action_range=action_range)

    mlp_args = {'hidden_layer_sizes': (8, 8),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'max_iter': 10000,
                'random_state': None,
                'tol': 0.0001,
                'verbose': False,
                'warm_start': True}

    deep_agent = DeepQLearning(mlp_args, epsilon=0.5, discount_factor=0.95, batch_size=1500, n_actions=10,
                                observation_range=observation_range,
                                action_range=action_range)

    agent = q_agent
    learning(agent, verbose=1000, n_episodes=10000)
    agent.save()
