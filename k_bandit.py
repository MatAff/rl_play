#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt

# Single bandit
class bandit:
	
	def __init__(self):
		self.mu = random.random() # Uniform distribution
		
	def pull(self):
		return(random.gauss(self.mu, 1))

# Multiple bandits		
class k_bandit:
	
	def __init__(self, k):
		self.bandits = [bandit() for i in range(k)]
		
	def pull(self, i):
		return(self.bandits[i].pull())

# Create the env
k = 10
bandits = k_bandit(k)

# Settings
nr_iter = 10000
alpha = 0.1
	
# Averages
avgs = np.zeros(k)
counts = np.ones(k)
overall_avg = 0
overall_count = 0
rec_avg = np.zeros(nr_iter)

# Iterate
for i in range(nr_iter):
	if random.random() < alpha:
		action = random.randint(0,k-1)
	else:
		action = np.argmax(avgs)

	# Apply action
	r = bandits.pull(action)
	
	# Update averages
	avgs[action] = ((avgs[action] * counts[action]) + r) / (counts[action] + 1)
	counts[action] += 1
	
	# Update overall average
	overall_avg = ((overall_avg * overall_count) + r) / (overall_count + 1)
	overall_count += 1
	
	# Track overall average
	rec_avg[i] = overall_avg
	

plt.plot(rec_avg)