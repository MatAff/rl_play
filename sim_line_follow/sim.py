#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import keras

# Keys
ESC_KEY = 27
KEY_P = 112

# A point should be an np.array with shape (2,)

class Line(object):
	
	def __init__(self, p1, p2):	
		self.points = np.array([p1, p2])
		
	def get_point(self, pos):
		return(self.points[pos,:])

    # Intersect last line segment with first line segment
	def intersect(self, other):				
		CA = self.points[-2,:] - other.points[0,:]
		AB = self.points[-1,:] - self.points[-2,:]
		CD = other.points[1,:] - other.points[0,:]				
		denom = np.cross(CD, AB) 
		if denom != 0:
			s = np.cross(CA,AB) / denom 
			i  = other.points[0,:] + s * CD
			overlap = self.in_range(i) and other.in_range(i)
			return(i, True, overlap)
		else:
			return None, False, False
		
	def in_range(self, p):
		return(((self.points[0,:] <= p)==(p <= self.points[1,:])).all())		
	
class Spline(object):
	
	def __init__(self, l1, l2, n):	
		self.points = np.zeros((n,2))			
		B = l1.get_point(-1)
		C = l2.get_point(0)
		I, _, _ = l1.intersect(l2)				
		for i in range(n):
			ratio = i / (n-1)
			S = self.rel_line(B, I, ratio)
			E = self.rel_line(I, C, ratio)
			P = self.rel_line(S, E, ratio)
			self.points[i,:] = P	

	def rel_line(self, S, E, ratio):
		return(S + (E - S) * ratio)

class Recorder(object):
	
	def __init__(self, filename, fps, size):
		self.out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
	
	def write(self, frame):
		self.out.write(frame)
		
	def save_img(self, frame, filename=None):
		if filename is None : filename = str(time.time()) + '.png'
		cv2.imwrite(filename, frame)
		
	def release(self):
		self.out.release()

class Course(object):
	
	def __init__(self):		
		line_list = []		
		sections = [Line(np.array([0,0]), np.array([1,0])),
				    Line(np.array([5,5]), np.array([4,6.5])),
			        Line(np.array([-4,6]), np.array([-5,5]))]

		for i, sect in enumerate(sections):
			spline = Spline(sections[i-1], sect, 25)
			line_list.append(spline.points)
			line_list.append(sect.points)

		self.points = np.concatenate(line_list)			

	def draw(self, frame):
		draw_line(frame, self.points, (255,0,0))

# Function to convert cartesian coordinate into pixel coordinates
def to_pixel(cart):	 
	S = np.array([scale, -scale])
	T = np.array([320, 480 - 80])
	return(tuple((cart * S + T).astype(int)))

def draw_line(frame, points, color):
	for i in range(points.shape[0] - 1):		
		s_pix = to_pixel(points[i+0,:])
		e_pix = to_pixel(points[i+1,:])
		try:
			cv2.line(frame, s_pix, e_pix,color,2)		
		except Exception as e:
			print(points)
			print(s_pix, e_pix)
			raise e
	
def rotation(rad):
	return(np.matrix([[math.cos(rad), -math.sin(rad)],
					  [math.sin(rad),  math.cos(rad)]]))	

class Car(object):
	
	def __init__(self):
		self.pos = np.array([[0],[0]])		
		self.dir = np.array([[1],[0]])
		
	def move(self, x, rad):
		self.pos = self.pos + x * self.dir * 0.5
		self.dir = np.matmul(rotation(rad), self.dir)
		self.pos = self.pos + x * self.dir * 0.5
	
	def draw(self, frame):
		tl = self.pos + np.matmul(rotation(math.pi *  0.5), self.dir * 0.3)
		tr = self.pos + np.matmul(rotation(math.pi * -0.5), self.dir * 0.3)
		bl = self.pos - self.dir + np.matmul(rotation(math.pi *  0.5), self.dir * 0.3)
		br = self.pos - self.dir + np.matmul(rotation(math.pi * -0.5), self.dir * 0.3)		
		points = np.array(np.concatenate([tl, tr, br, bl, tl, br, bl, tr], axis=1).transpose())				
		draw_line(frame, points, (0,0,255))			
	
	def detect(self, points, frame, dist):
		
		# Create detection line
		sp = self.pos + np.matmul(rotation(math.pi * -0.25), self.dir * dist)
		ep = self.pos + np.matmul(rotation(math.pi *  0.25), self.dir * dist)
		sp = np.squeeze(np.array(sp)) 
		ep = np.squeeze(np.array(ep)) 
		detect_line = Line(sp, ep)
		
		# Draw detection line
		draw_line(frame, detect_line.points, (0,200,0))
				
		# Loop through course
		for i in range(points.shape[0] - 1):
			sub_line = Line(points[i,:], points[i+1,:])
			inters, has_intersect, overlap = detect_line.intersect(sub_line)			
			if overlap == True: 	
				pos = np.linalg.norm(inters - sp) / np.linalg.norm(ep - sp)
				return((pos - 0.5) * 2.0)
	
	def detect_list(self, points, frame, dist_list):
		return([self.detect(points, frame, dist) for dist in dist_list])

class Retainer(object):
	
	def __init__(self, data):
		self.prev = data
		
	def retain(self, current):
		current = current.astype('float')
		current[np.isnan(current)] = self.prev[np.isnan(current)]
		self.prev = current
		return(current)		

def discount_reward(rewards, discount):
	running_reward = 0.0
	discounted_rewards = np.array([], 'float')
	for i in reversed(range(rewards.shape[0])):
		discounted_rewards = np.concatenate(([running_reward], discounted_rewards))
		running_reward = running_reward * discount + rewards[i] * (1 - discount)	
	return(discounted_rewards)

def create_model(shape):
	activation = 'relu'
	model = keras.Sequential()
	model.add(keras.layers.Dense(10, activation=activation, input_shape=(shape[1],)))
	model.add(keras.layers.Dense(20, activation=activation))
	model.add(keras.layers.Dense(20, activation=activation))
	model.add(keras.layers.Dense(1))
	model.compile(optimizer='rmsprop', loss='mse')
	return(model)

def select_reverse(states, actions, rewards):

	# Create and fit model
	model = create_model(states.shape)
	model.fit(states, rewards, epochs=10, batch_size=256, verbose=0) 

    # Predict expected rewards
	expected_rewards = model.predict(states)
	plt.scatter(rewards, expected_rewards)
	plt.show()
		
	# Select better than expected performing data
	good_cases = (rewards > expected_rewards)[:,0]
	return(states[good_cases,:], actions[good_cases])	

def select_data(states, actions, rewards):

	if states.shape[0] != actions.shape[0] or states.shape[0] != rewards.shape[0] : raise ValueError('Shapes do not align')
	
	# Create and fit model
	model = create_model(states.shape)
	model.fit(states, rewards, epochs=20, batch_size=256, verbose=0) 

    # Predict expected rewards
	expected_rewards = model.predict(states)
	expected_rewards = np.squeeze(expected_rewards)
	good_cases = (rewards < 0.75 * expected_rewards)
	print(np.unique(good_cases, return_counts=True))
	
	# Plot
	plt.scatter(expected_rewards, rewards, c=good_cases)
	plt.plot([0, 3], [0, 3])
	plt.xlabel('expected rewards')
	plt.ylabel('rewards')
	plt.show()
			
	return(states[good_cases,:], actions[good_cases])

def select_by_episode(states, actions, mean_rewards):
	med = np.median(mean_rewards)
	return(states[mean_rewards < med,:], actions[mean_rewards < med])

# Generic RI control class not specific to task at hand
class ControlBase(object):
	
	def __init__(self, state_space):
		self.state_space = state_space
		self.all_states = np.empty((0, state_space))
		self.all_actions = np.empty((0, 1))
		self.all_rewards = np.empty((0, 1))	

	def pre(self):
		self.states = np.empty((0,self.state_space))
		self.actions = np.empty((0,1))
		self.rewards = np.empty((0,1))
		
	def post(self):
		self.shift = 1
		self.all_states = np.append(self.all_states, self.states[0:-self.shift,:], 0)
		self.all_actions = np.append(self.all_actions, self.actions[0:-self.shift])
		self.all_rewards = np.append(self.all_rewards, self.rewards[self.shift:])		

# RI Control class dedicated to task
class ControlLineFollow(ControlBase):
	
	def __init__(self, state_space):
		super(ControlLineFollow, self).__init__(state_space)
		self.phase = 0
		self.all_discounted_rewards = np.empty((0,1))
		self.all_mean_rewards = np.empty((0,1))
		self.mean_reward_list = np.empty((0,1))			
				
	def pre(self, run_nr): 		
		super(ControlLineFollow, self).pre() # Run super					
		self.ret = Retainer(np.zeros(len(dist_list)))		# Set retrainer
		self.err = 0.0 # Set error		
		self.err_discount = 0.9
				
	def post(self, run_nr): 		
		super(ControlLineFollow, self).post()  # Run super
		self.phase = min(self.phase + 1, 2) # Update phase
		
		# Mean reward
		mean_reward = np.mean(self.rewards)
		print('Mean reward (less is better): %.2f' % mean_reward)
		self.all_mean_rewards = np.append(self.all_mean_rewards, np.repeat(mean_reward, len(self.rewards) - 1))
		self.mean_reward_list = np.append(self.mean_reward_list, mean_reward)
		
		# Discounted reward
		discounted_rewards = discount_reward(self.rewards, 0.9)
		self.all_discounted_rewards = np.append(self.all_discounted_rewards, discounted_rewards[self.shift:])
				
	def pos_to_reward(self, line_pos):
		pos = np.array(line_pos).astype('float')
		pos[np.isnan(pos)] = 5.0
		return np.abs(pos)[0]
				
	def decide(self, line_pos):
		self.rewards = np.append(self.rewards, self.pos_to_reward(line_pos))  # Update reward list
		self.line_pos = self.ret.retain(np.array(line_pos)) # Insert retained values
		self.err = self.err * self.err_discount + (1 - self.err_discount) * ((random.random() - 0.5) / 30.0) # Update error
		self.states = np.append(self.states, np.array([self.line_pos]), 0) # Store

# RI control class based on subsetting learning data
class ControlSubData(ControlLineFollow):
	
	def __init__(self, state_space):
		super(ControlSubData, self).__init__(state_space)
	
	def pre(self, run_nr):
		super(ControlSubData, self).pre(run_nr)
		
		if run_nr == 1:
			self.model = create_model(self.all_states.shape)
			self.model.fit(self.all_states, self.all_actions, epochs=25, batch_size=256, verbose=0)
		if run_nr > 1:					
			self.model = create_model(self.all_states.shape)
			sub_states, sub_actions = select_data(self.all_states, self.all_actions, self.all_discounted_rewards)
			#sub_states, sub_actions = select_reverse(self.all_states, self.all_actions, self.all_rewards)
			#sub_states, sub_actions = select_by_episode(self.all_states, self.all_actions, self.all_mean_rewards)
			#sub_states, sub_actions = select_by_episode(self.all_states, self.all_actions, self.all_rewards)  
			self.model.fit(sub_states, sub_actions, epochs=25, batch_size=256, verbose=0)

	def post(self, run_nr):
		super(ControlSubData, self).post(run_nr)

	def decide(self, line_pos):
		super(ControlSubData, self).decide(line_pos)		
		if self.phase == 0:
			if self.line_pos[1] is not None: 
				rotate = self.line_pos[1] * 0.1
		else:
			rotate = self.model.predict(np.array([self.line_pos]))[0,0]		
		rotate += self.err
		self.actions = np.append(self.actions, rotate)
		return(rotate)

# RI control class based on subsetting learning data
class ControlStateAction(ControlLineFollow):
	
	def __init__(self, state_space):
		super(ControlStateAction, self).__init__(state_space)
	
	def pre(self, run_nr):
		super(ControlStateAction, self).pre(run_nr)
		
		if run_nr == 1:					
			all_states_actions = np.append(self.all_states, np.array([self.all_actions]).transpose(), axis=1)
			self.model = create_model(all_states_actions.shape)
			self.model.fit(all_states_actions, control.all_discounted_rewards, epochs=100, batch_size=256, verbose=0)

	def post(self, run_nr):
		super(ControlStateAction, self).post(run_nr)

	def decide(self, line_pos):
		super(ControlStateAction, self).decide(line_pos)		
		if self.phase == 0:
			if self.line_pos[1] is not None: 
				self.rotate = self.line_pos[1] * 0.1
		else:
			pos_actions = np.arange(-0.1, 0.1, 0.01)
			val = np.empty((0,1))
			for act in pos_actions:
				X = np.append(line_pos, act)
				y = control.model.predict(np.array([X]))[0]
				val = np.append(val, y)
			plt.plot(val)
			best_act = pos_actions[np.argmin(val)]
			print(best_act)
			self.rotate = best_act

		self.rotate += self.err
		self.actions = np.append(self.actions, self.rotate)
		return(self.rotate)
		
####################
### MAIN SECTION ###
####################
	
# Display settings
frame_name = 'Sim'
cv2.namedWindow(frame_name)		
height = 480
width = 640
scale = 35

# Global red
glob_rec = Recorder('RL.avi', 30, (width, height))
rec_run = [0,1,2,5,10,20,50]

# Run settings
nr_runs = 1000
frames_per_run = 500
running = True

# Instantiate sim and control elements
course = Course()
dist_list = [0.5, 1.0, 1.5, 2.0, 2.5]
control = ControlSubData(len(dist_list))		
#control = ControlStateAction(len(dist_list))		

# Loop through runs
for run in range(nr_runs):
	
	# Reset
	print(run)
	car = Car()	
	
	# Control
	control.pre(run)
	
	# Recored
	filename = './run_%03i.avi' % run 
	rec = Recorder(filename, 30, (width, height))
	
	# Main loop
	for frame_nr in range(frames_per_run):
		
		try:
		
			# Plot course and car
			frame = np.zeros((height, width,3), np.uint8)	
			course.draw(frame)
			car.draw(frame)
			
			# Detect and plot detect
			line_pos = car.detect_list(course.points, frame, dist_list)
			
			# Show 
			frame = cv2.putText(frame, 'Run %i' % run, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1) 
			rec.write(frame)
			if run in rec_run : glob_rec.write(frame)
			cv2.imshow(frame_name, frame)
			key = cv2.waitKey(5)
			if key != -1 : print(key)
			
			# Process key
			if key == ESC_KEY : raise ValueError('ESC pressed')	
			if key == KEY_P : rec.save_img(frame)
				
			# Decide
			rotate = control.decide(line_pos)
			
			# Act			
			car.move(0.2, rotate)
			
		except Exception as e: 
			
			print(e)		
			running = False
			break

    # Control
	control.post(run)
	
	# Plot
	plt.plot(control.mean_reward_list)
	plt.show()
	
	# Recorder
	rec.release()	
	glob_rec.release()
	
	if running == False : break
			
cv2.destroyWindow(frame_name)	


# Save models
# Set error as ratio of action variance

# Add key to kill whole process >> Done
# Keep frame alive >> Done
# Create model only once >> Done
# Speed up running process >> Done
# Retain previous position when None >> Done
# Move RI related code to class >> Done
# Separate storing actions and state >> Done
# Explore discount rate of reward >> Done
# Explore penaly of None >> Done
# Add recorder >> Done


