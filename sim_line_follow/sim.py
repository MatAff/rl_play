#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
import keras

# Keys
ESC_KEY = 27

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

# Course class
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
	T = np.array([320, 480 - 40])
	return(tuple((cart * S + T).astype(int)))

def draw_line(frame, points, color):
	for i in range(points.shape[0] - 1):		
		s_pix = to_pixel(points[i+0,:])
		e_pix = to_pixel(points[i+1,:])
		cv2.line(frame, s_pix, e_pix,color,2)		
	
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

def create_model(shape):
	model = keras.Sequential()
	model.add(keras.layers.Dense(10, activation='relu', input_shape=(shape[1],)))
	model.add(keras.layers.Dense(10, activation='relu'))
	model.add(keras.layers.Dense(1))
	model.compile(optimizer='rmsprop', loss='mse')
	print(model.summary())
	return(model)

def assign_reward(rewards, discount):
	running_reward = 0.0
	discounted_rewards = np.array([], 'float')
	for i in reversed(range(rewards.shape[0])):
		discounted_rewards = np.concatenate(([running_reward], discounted_rewards))
		running_reward = running_reward * discount + rewards[i] * (1 - discount)	
	return(discounted_rewards)

def select_data(action_state, rewards):

	# Preprocess data
	X = action_state[:,1:]	
	y = rewards
	
	# Create and fit model
	model = create_model(X.shape)
	model.fit(X, y, epochs=50, batch_size=256, verbose=0) 

    # Predict expected rewards
	expected_rewards = model.predict(X)
	plt.scatter(rewards, expected_rewards)
		
	# Select better than expected performing data
	sub_action_state = action_state[rewards < expected_rewards[:,0],:]
	
	return(sub_action_state)

class Control(object):
	
	def __init__(self):
		self.phase = 0
		self.ret = Retainer(np.zeros(len(dist_list)))	
		
		# Create place to store action state
		self.action_state = np.empty((0,len(dist_list) + 1))
		self.rewards = np.empty((0,1))

	def pos_to_reward(self, line_pos):
		pos = np.array(line_pos).astype('float')
		pos[np.isnan(pos)] = 3.0
		return np.abs(pos)[0]
				
	def decide(self, line_pos):
		
		# Update reward list
		self.rewards = np.append(self.rewards, self.pos_to_reward(line_pos)) 
		
		# Insert retained values
		line_pos = self.ret.retain(np.array(line_pos)) 
				
		if self.phase == 0:
			if line_pos[1] is not None: 
				rotate = line_pos[1] * 0.1
		else:
			rotate = self.model.predict(np.array([line_pos]))[0,0]				
			self.err = self.err * self.err_discount + (1 - self.err_discount) * ((random.random() - 0.5) / 10.0)
			rotate += self.err
		
		# Store
		store_arr = np.concatenate(([rotate],line_pos))
		self.action_state = np.append(self.action_state, np.array([store_arr]), 0)
							
		return(rotate)
	
	def next(self): # Rename this! Poor name choice!
		self.phase = min(self.phase + 1, 2)

		self.err = 0.0
		self.err_discount = 0.9

		
		if self.phase ==1:
			
			# Randomness # Should move to controller
			
			X = self.action_state[:,1:]
			y = self.action_state[:,0]
			
			# Create model
			self.model = create_model(X.shape)

			# Fit model
			self.model.fit(X, y, epochs=50, batch_size=256, verbose=0)
		
		if self.phase > 1:
			# Select data		
			values = assign_reward(self.rewards, 0.75)
			sub_action_state = select_data(self.action_state, values)

			# Preprocess data	
			X = sub_action_state[:,1:]
			y = sub_action_state[:,0]					
				
			# Fit model
			self.model.fit(X, y, epochs=50, batch_size=256, verbose=0)

####################
### MAIN SECTION ###
####################
	
# Display settings
frame_name = 'Sim'
cv2.namedWindow(frame_name)		
height = 480
width = 640
scale = 35

# Run settings
nr_runs = 100
frames_per_run = 400
running = True

# Instantiate sim and control elements
course = Course()
dist_list = [0.5, 1.0, 1.5, 2.0, 2.5]
control = Control()		

# Loop through runs
for run in range(nr_runs):
	
	# Reset
	print(run)
	car = Car()	
	
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
			cv2.imshow(frame_name, frame)
			key = cv2.waitKey(5)
			
			# Process key
			if key == ESC_KEY : raise ValueError('ESC pressed')	
				
			# Decide
			rotate = control.decide(line_pos)
			
			# Act
			car.move(0.2, rotate)
			
		except Exception as e: 
			
			print(e)		
			running = False
			break

	control.next()
	
	if running == False : break
			
cv2.destroyWindow(frame_name)			

	
# Separate storing actions and state
# Explore discount rate of reward
# Explore penaly of None
# Set error as ratio of action variance

# Add key to kill whole process >> Done
# Keep frame alive >> Done
# Create model only once >> Done
# Speed up running process >> Done
# Retain previous position when None >> Done
# Move RI related code to class >> Done
			
	
