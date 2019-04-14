#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt

# Single bandit
class bandit(object):
	
	def __init__(self):
		self.mu = random.random() # Uniform distribution
		
	def pull(self):
		return(random.gauss(self.mu, 1))

# Multiple bandits		
class k_bandit(object):
	
	def __init__(self, k):
		self.bandits = [bandit() for i in range(k)]
		
	def get_action_space(self):
		return(len(self.bandits))

	def pull(self, i):
		return(self.bandits[i].pull())
		
	def mu(self):
		return[b.mu for b in self.bandits]

# Action values
class action_value(object):

	def __init__(self, action_space):
		self.Q_action = np.zeros(action_space)
		self.n_action = np.ones(action_space)

	def update(self, action, r):
		self.Q_action[action] = ((self.Q_action[action] * self.n_action[action]) + r) / (self.n_action[action] + 1)
		self.n_action[action] += 1
	
# Total reward
class total_reward(object):
	
	def __init__(self):
		self.R_avg = 0
		self.n = 0
		self.R_avg_track = [0]

	def update(self, r):
		self.R_avg = ((self.R_avg * self.n) + r) / (self.n + 1)
		self.n += 1
		self.R_avg_track.extend([self.R_avg]) # Track overall average

# Function epsilon greedy
def epsilon_greedy(Q_action, epsilon):
	if random.random() < epsilon:
		action = random.randint(0,k-1)
	else:
		action = np.argmax(Q_action)
	return action
	
def run_epsilon_greedy(nr_steps, epsilon):

	Q = action_value(bandits.get_action_space()) # Action value
	R = total_reward() # Total reward

	for i in range(nr_steps):
		
		# Determine action
		action = epsilon_greedy(Q.Q_action, epsilon)

		# Perform action
		r = bandits.pull(action)
		
		# Update action values
		Q.update(action, r)
		
		# Update overall rewards
		R.update(r)
		
	return(R.R_avg_track)

# Create the env
k = 10
bandits = k_bandit(k)
bandits.mu()

# Settings
nr_steps = 1000
epsilon = 0.1

# Single run
track = run_epsilon_greedy(nr_steps, epsilon)
plt.plot(track)

# Multiple runs
nr_runs = 1000
multi_track = []
for _ in range(nr_runs):
	track = run_epsilon_greedy(nr_steps, epsilon)
	multi_track.extend([track])

# Show aggregate reward
np_track = np.stack(multi_track)
avg_track = np_track.mean(axis=0)
plt.plot(avg_track)

### TRY DIFFERENT EPSILON

epsilon = 0

# Multiple runs
nr_runs = 1000
multi_track = []
for _ in range(nr_runs):
	track = run_epsilon_greedy(nr_steps, epsilon)
	multi_track.extend([track])

# Show aggregate reward
np_track = np.stack(multi_track)
avg_track = np_track.mean(axis=0)
plt.plot(avg_track)


