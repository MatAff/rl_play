#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random

# Keys
ESC_KEY = 27

# A point should be an np.array with shape [2,]

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
				return(np.linalg.norm(inters - sp) / np.linalg.norm(ep - sp))
	
	def detect_list(self, points, frame, dist_list):
		return([self.detect(points, frame, dist) for dist in dist_list])
	
# Instantiate course and car		
course = Course()
car = Car()
dist_list = [0.5, 1.0, 1.5, 2.0, 2.5]

# Create window
frame_name = 'Sim'
cv2.namedWindow(frame_name)		

# Frame settings
height = 480
width = 640
scale = 35

# Break variable
running = True
counter = 0

# Store measurement and action
action_state = np.empty((0,len(dist_list) + 1))

# Main loop
while running:
	
	try:
	
		# Create frame and draw elements
		frame = np.zeros((height, width,3), np.uint8)	
		course.draw(frame)
		car.draw(frame)
		
		# Detect 		
		line_pos = car.detect_list(course.points, frame, dist_list)
		
		# Show and get key
		cv2.imshow(frame_name, frame)
		key = cv2.waitKey(20)
	
		# Process key
		if key == ESC_KEY:
			running = False
			
		# Counter
		counter += 1
		if counter > 1000:
			running = False
		
		# Decide
		if line_pos[1] is not None: 
			rotate = (line_pos[1] - 0.5) * 0.2
			
		# Store 
		store_arr = np.concatenate(([rotate],line_pos))
		action_state = np.append(action_state, np.array([store_arr]), 0)
	
		# Action 
		car.move(0.2, rotate)
		
	except Exception as e: 
		
		print(e)		
		running = False
		
cv2.destroyWindow(frame_name)			

### MACHINE LEARNING SECTION ###

import keras

def preprocess(X):
	X = X.astype('float')
	X = X - 0.5
	X = np.nan_to_num(X)
	return(X)

def create_model(shape):
	model = keras.Sequential()
	model.add(keras.layers.Dense(25, activation='relu', input_shape=(shape[1],)))
	model.add(keras.layers.Dense(25, activation='relu'))
	model.add(keras.layers.Dense(25, activation='relu'))
	model.add(keras.layers.Dense(1))
	model.compile(optimizer='rmsprop', loss='mse') # For regression
	print(model.summary())
	return(model)

print(action_state.shape)
X = action_state[:,1:]
y = action_state[:,0]

# Preprocess data
X = preprocess(X)

# Create model
model = create_model(X.shape)

# Fit model
model.fit(X, y, epochs=500, batch_size=256, verbose=0)

x = preprocess(np.array([line_pos]))
model.predict(x)

# Recreate frame
cv2.namedWindow(frame_name)		

# Reset car
car = Car()

# Break variable
running = True
counter = 0

# Main loop
while running:
	
	try:
	
		# Create frame and draw elements
		frame = np.zeros((height, width,3), np.uint8)	
		course.draw(frame)
		car.draw(frame)
		
		# Detect 
		line_pos = car.detect_list(course.points, frame, dist_list)
		
		# Show and get key
		cv2.imshow(frame_name, frame)
		key = cv2.waitKey(20)
	
		# Process key
		if key == ESC_KEY:
			running = False
			
		# Counter
		counter += 1
		if counter > 1000:
			running = False
		
		# Decide
		x = preprocess(np.array([line_pos]))
		rotate = model.predict(x)[0,0]	

		# Store 
		store_arr = np.concatenate(([rotate],line_pos))
		action_state = np.append(action_state, np.array([store_arr]), 0)	

		# Action 
		car.move(0.2, rotate)
		
	except Exception as e: 
		
		print(e)		
		running = False
		
cv2.destroyWindow(frame_name)		

print(action_state.shape)

def assign_reward(action_state, discount):
	position = action_state[:,1]
	position = position.astype('float')
	position = position - 0.5	
	position[np.isnan(position)] = 2.0
	position = np.absolute(position)
	position[0:100]
	running_reward = 0.0
	rewards = np.array([], 'float')
	for i in reversed(range(position.shape[0])):
		rewards = np.concatenate(([running_reward], rewards))
		running_reward = running_reward * discount + position[i] * (1 - discount)
	return(rewards)

def select_data(action_state, rewards):

	# Preprocess data
	X = action_state[:,1:]	
	X = preprocess(X)
	y = rewards
	
	print(X.shape)
	print(y.shape)
	
	# Create and fit model
	model = create_model(X.shape)
	model.fit(X, y, epochs=50, batch_size=256, verbose=0) # Less training to avoid overfitting

    # Predict expected rewards
	expected_rewards = model.predict(X)
	plt.scatter(rewards, expected_rewards)
		
	# Select better than expected performing data
	sub_action_state = action_state[rewards < expected_rewards[:,0],:]
	print(sub_action_state.shape)
	
	return(sub_action_state)
	
for j in range(1000):

	rewards = assign_reward(action_state, 0.99)
	
	plt.plot(rewards)
	
	# Select data
	sub_action_state = select_data(action_state, rewards)
	
	# Select trainig cases
	#	med = np.median(rewards)
	#	#med = 2.0 # Override
	#	sub_action_state = action_state[rewards < med,:]
	#	print(sub_action_state.shape)

	# Preprocess data	
	X = sub_action_state[:,1:]
	y = sub_action_state[:,0]
	X = preprocess(X)
	
	# Create model
	model = create_model(X.shape)
	
	# Fit model
	model.fit(X, y, epochs=500, batch_size=256, verbose=0)
	
	# Recreate frame
	cv2.namedWindow(frame_name)		
	
	# Reset car
	car = Car()

	# Break variable
	running = True
	counter = 0
	
	# Randomness
	err = (random.random() - 0.5) /10
	err_discount = 0.9
	
	# Main loop
	while running:
		
		try:
		
			# Create frame and draw elements
			frame = np.zeros((height, width,3), np.uint8)	
			course.draw(frame)
			car.draw(frame)
			
			# Detect 
			line_pos = car.detect_list(course.points, frame, dist_list)
			
			# Show and get key
			cv2.imshow(frame_name, frame)
			key = cv2.waitKey(20)
		
			# Process key
			if key == ESC_KEY:
				running = False
			
			# Counter
			counter += 1
			if counter > 1000:
				running = False
			
			# Decide
			x = preprocess(np.array([line_pos]))
			rotate = model.predict(x)[0,0]	
			
			# Add error
			err = err * err_discount + (1- err_discount) * ((random.random() - 0.5) / 10)
			print(err)
			print(rotate)
			rotate += err
	
			# Store 
			store_arr = np.concatenate(([rotate],line_pos))
			action_state = np.append(action_state, np.array([store_arr]), 0)	
	
			# Action 
			car.move(0.2, rotate)
			
		except Exception as e: 
			
			print(e)		
			running = False
			
	cv2.destroyWindow(frame_name)		
