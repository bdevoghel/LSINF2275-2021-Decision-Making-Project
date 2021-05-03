import gym  # doc available here : https://gym.openai.com/docs/

env = gym.make('MountainCarContinuous-v0')
"""
From https://gym.openai.com/envs/MountainCarContinuous-v0/ : A car is on a one-dimensional track, positioned 
between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong 
enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build 
up momentum. Here, the reward is greater if you spend less energy to reach the goal 
"""

env.reset()
done = False

print(f"Limits of observation space (position, speed)                   : low={env.observation_space.low}, high={env.observation_space.high}")
print(f"Limits of action space (~acceleration)                          : low={env.action_space.low}, high={env.action_space.high}")
print(f"Reward range (reward is inversely proportional to spent energy) : {env.reward_range}")

n_episodes = 1
n_steps = 100

for i_episode in range(n_episodes):
    print(f"- Starting episode {i_episode + 1}/{n_episodes}")
    observation = env.reset()
    print(observation)
    for t in range(n_steps):
        env.render()
        action = env.action_space.sample()  # take a random action
        observation, reward, done, info = env.step(action)
        print(observation, reward)

        if done:
            print(f"Episode finished after {t+1} timesteps with reward={reward}")
            break
    print(f"- End of episode {i_episode + 1}/{n_episodes}")


env.close()
