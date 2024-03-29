import gym
from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from functools import partial
import numpy
# %matplotlib inline
plt.style.use('ggplot')


env = gym.make('Blackjack-v0')

""" 
Then we define the policy function which takes the current state and checks if the score is greater than or equal to 2o.

if it is, we return 0 or else we return 1. 

That is, if the score is greater than or equal to 20, we stand (0) or else we hit (1)

"""
env.action_space, env.observation_space
def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

"""
We define states, actions, and rewards as a list and initiate the environment using env.reset 
store an observation variable.

until we reach the terminal state, that is, till the end of the episode, we do the following:

    1. Append the observation to the states list.
               
    2. Now, we create an action using our sample_policy function and append the actions to an action list.
    
    3. Then, for each step in the environment, we store the state, reward, and done 
    (which specifies whether we reached terminal state) and we append the rewards to the reward list.

    4. If we reached the terminal state, then we break.
"""
def generate_episode(policy, env):
    states, actions, rewards = [], [], []
    observation = env.reset()
    while True:
        states.append(observation)
        action = sample_policy(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break
    return states, actions, rewards


"""
________________________

get the value of each state using the first visit Monte Carlo method. 
________________________

First, we initialize the empty value table as a dictionary for storing the values of each state. 

Then, for a certain number of episodes, we do the following: 

    1. First, we generate an episode and store the states and rewards; we initialize returns as 0 which is the sum of rewards

    2. Then for each step, we store the rewards to a variable R and states to S, and 
    we calculate returns as a sum of rewards.

    3. We now perform the first visit Monte Carlo; we check if the episode is being visited for the visit time. 
    If it is, we simply take the average of returns and assign the value of the state as an average of returns.
"""

def first_visit_mc_prediction(policy, env, n_episodes):
    value_table = defaultdict(float)
    N = defaultdict(int)
    for _ in range(n_episodes):
        states, _, rewards = generate_episode(policy, env)
        returns = 0
        for t in range(len(states) - 1, -1, -1):
            R = rewards[t]
            S = states[t]
            returns += R
            if S not in states[:t]:
                N[S] += 1
                value_table[S] += (returns - value_table[S]) / N[S]
    return value_table

value = first_visit_mc_prediction(sample_policy, env, n_episodes=500000)

def plot_blackjack(V, ax1, ax2):
    player_sum = numpy.arange(12, 21 + 1)
    dealer_show = numpy.arange(1, 10 + 1)
    usable_ace = numpy.array([False, True])
    state_values = numpy.zeros((len(player_sum),
                                len(dealer_show),
                                len(usable_ace)))
    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = V[player, dealer, ace]
    X, Y = numpy.meshgrid(player_sum, dealer_show)
    ax1.plot_wireframe(X, Y, state_values[:, :, 0])
    ax2.plot_wireframe(X, Y, state_values[:, :, 1])
    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('player sum')
        ax.set_xlabel('dealer showing')
        ax.set_zlabel('state-value')


fig, axes = pyplot.subplots(nrows=2, figsize=(5, 8),
subplot_kw={'projection': '3d'})
axes[0].set_title('value function without usable ace')
axes[1].set_title('value function with usable ace')
plot_blackjack(value, axes[0], axes[1])
plt.show()



