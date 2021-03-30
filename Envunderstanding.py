import gym
import numpy as np
env = gym.make('FrozenLake-v0')

print(env.action_space.n)