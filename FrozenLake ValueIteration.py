import gym
import numpy as np 

env = gym.make('FrozenLake-v0')

""" 
    First, we initialize the random value table which is 0 for all the states and numbers of iterations:


    Then, upon starting each iteration, we copy the value_table to updated_value_table:

        Now we calculate the Q table and pick up the maximum state-action pair which has the highest value as the value table.

        Instead of creating a Q table for each state, we create a list called Q_value, 
        then for each action in the state, we create a list called next_states_rewards,
        which store the Q_value for the next transition state. 

        Then we sum the next_state_rewards and append it to our Q_value.

    Then, we will check whether we have reached the convergence, that is, 
    the difference between our value table and updated value table is very small.

""" 


def value_iteration(env, gamma = 0.6):
    value_table = np.zeros(env.observation_space.n)
    no_of_iterations = 100000
    threshold = 1e-20
    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_states_rewards = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state])))
                Q_value.append(np.sum(next_states_rewards))
                # print(Q_value[0:3])
            value_table[state] = max(Q_value)
            # print(value_table[0:3])
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return value_table, Q_value

optimal_value_function, optimal_Q_function = value_iteration(env=env,gamma=0.6)

"""
    After finding optimal_value_function, 
    how can we extract the optimal policy from the optimal_value_function? 

    We calculate the Q value using our optimal value action and 
    pick up the actions which have the highest Q value for each state as the optimal policy. 
    
    We do this via a function called extract_policy().

"""

""" 
Method : 
        First, we define the random policy; we define it as 0 for all the states 

        Then, for each state, we build a Q_table and for each action in that state we compute the Q value and 
        add it to our Q_table

        Then we pick up the policy for the state as the action that has the highest Q value

"""

def extract_policy(value_table, gamma = 1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        policy[state] = np.argmax(Q_table)
    return policy

optimal_policy = extract_policy(optimal_value_function, gamma=0.6)

print(optimal_policy)