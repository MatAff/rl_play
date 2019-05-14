#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt

# Single bandit
class bandit(object):
	
	def __init__(self):
		self.mu = random.gauss(0, 1) 
		
	def pull(self):
		return(random.gauss(self.mu, 1))
	
	def nonstationary(self, sd):
		self.mu = self.mu + random.gauss(0, sd)

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
		
	def nonstationary(self, sd):
		[b.nonstationary(sd) for b in self.bandits]

# Action values
class action_value(object):

	def __init__(self, action_space, stepsize=None):
		self.Q_action = np.zeros(action_space)
		self.N = np.zeros(action_space)
		self.stepsize = stepsize

	def update(self, action, r):
		if self.stepsize is not None:
			self.Q_action[action] = self.Q_action[action] + self.stepsize * (r - self.Q_action[action])  
		else:
			self.N += 1
			self.Q_action[action] = self.Q_action[action] + (1 / self.N[action]) * (r - self.Q_action[action])  
		
# Function epsilon greedy
def epsilon_greedy(Q_action, epsilon):
	if random.random() < epsilon:
		action = random.randint(0,k-1)
	else:
		action = np.argmax(Q_action)
	return action

def run_epsilon_greedy(k, n, epsilon, stepsize, nonstationary_sd):

	# Create bandits
	bandits = k_bandit(k)
	
	# Create action_value
	Q = action_value(bandits.get_action_space(), stepsize) # Action value
	R = np.empty((n))

	for step in range(n):
		
		# Determine action
		action = epsilon_greedy(Q.Q_action, epsilon)

		# Perform action
		r = bandits.pull(action)
		
		# Update action values
		Q.update(action, r)
		
		# Update overall rewards
		R[step] = r
		
		# Nonstationary
		bandits.nonstationary(nonstationary_sd)
		
	return(R)

# Parameters
k = 10
stepsize = 0.1
epsilon = 0.01
n = 10**3
nonstationary_sd = 0.01
reps = 1000

# Multiple runs
R_mat = np.empty((reps, n))
for rep in range(reps):
	R_mat[rep,] = run_epsilon_greedy(k, n, epsilon, stepsize, nonstationary_sd)	 

R_avg = np.mean(R_mat, axis=0)

# Plot
plt.plot(R_avg)

# Parameters
k = 10
stepsize = None
epsilon = 0.01
n = 10**3
nonstationary_sd = 0.01
reps = 1000

# Multiple runs
R_mat2 = np.empty((reps, n))
for rep in range(reps):
	R_mat2[rep,] = run_epsilon_greedy(k, n, epsilon, stepsize, nonstationary_sd)	 

R_avg2 = np.mean(R_mat2, axis=0)

# Plot
plt.plot(R_avg2)

plt.plot(R_avg)
plt.plot(R_avg2)

## Single run
#R = run_epsilon_greedy(k, n, epsilon, stepsize, nonstationary_sd)
#plt.plot(R)

