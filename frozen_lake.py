#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import numpy as np
import time

env = gym.make('FrozenLake-v0')

# Explore state and action space
print(env.observation_space)
print(env.action_space)

def value_iteration(env, gamma = 1.0):
    
    # initialize value table with zeros
    value_table = np.zeros(env.observation_space.n)
    
    # set number of iterations and threshold
    no_of_iterations = 100000
    threshold = 1e-20
    
    for i in range(no_of_iterations):        
        updated_value_table = np.copy(value_table) 
               
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_states_rewards = []
                for next_sr in env.unwrapped.P[state][action]: 
                    trans_prob, next_state, reward_prob, _ = next_sr 
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * updated_value_table[next_state]))) 
                
                Q_value.append(np.sum(next_states_rewards))
                
            value_table[state] = max(Q_value) 
            
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
             print ('Value-iteration converged at iteration# %d.' %(i+1))
             break
    
    return value_table

def extract_policy(value_table, gamma = 1.0):
 
    # initialize the policy with zeros
    policy = np.zeros(env.observation_space.n) 
        
    for state in range(env.observation_space.n):
        
        # initialize the Q table for a state
        Q_table = np.zeros(env.action_space.n)
        
        # compute Q value for all ations in the state
        for action in range(env.action_space.n):
            for next_sr in env.unwrapped.P[state][action]: 
                trans_prob, next_state, reward_prob, _ = next_sr 
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        
        # select the action which has maximum Q value as an optimal action of the state
        policy[state] = np.argmax(Q_table)
    
    return policy
	
optimal_value_function = value_iteration(env=env, gamma=1.0)
optimal_policy = extract_policy(optimal_value_function, gamma=1.0)

print(optimal_policy)

env.reset()
env.render()
for a in optimal_policy:
	print(a)
	time.sleep(0.5)
	env.step(a)
	env.render()	


	


# Explore lake
for state in range(env.observation_space.n):
    for action in range(env.action_space.n):
        print(env.unwrapped.P[state][action]) 