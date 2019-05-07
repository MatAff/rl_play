#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

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
		sp = np.squeeze(np.array(self.pos))		
		ep = np.squeeze(np.array(self.pos - self.dir))	
		draw_line(frame, Line(sp, ep).points, (0,0,255))		
	
	def detect(self, points, frame):
		
		# Create detection line
		sp = self.pos + np.matmul(rotation(math.pi * -0.25), self.dir)
		ep = self.pos + np.matmul(rotation(math.pi *  0.25), self.dir)
		sp = np.squeeze(np.array(sp)) 
		ep = np.squeeze(np.array(ep)) 
		detect_line = Line(sp, ep)
		
		# Draw detection line
		draw_line(frame, detect_line.points, (0,255,0))
				
		# Loop through course
		for i in range(points.shape[0] - 1):
			sub_line = Line(points[i,:], points[i+1,:])
			inters, has_intersect, overlap = detect_line.intersect(sub_line)			
			if overlap == True: 	
				return(np.linalg.norm(inters - sp) / np.linalg.norm(ep - sp))

# Instantiate course and car		
course = Course()
car = Car()

# Create window
frame_name = 'Sim'
cv2.namedWindow(frame_name)		

# Frame settings
height = 480
width = 640
scale = 35

# Break variable
running = True

# Main loop
while running:
	
	try:
	
		# Create frame and draw elements
		frame = np.zeros((height, width,3), np.uint8)	
		course.draw(frame)
		car.draw(frame)
		
		# Detect 
		line_pos = car.detect(course.points, frame)
	
		# Show and get key
		cv2.imshow(frame_name, frame)
		key = cv2.waitKey(20)
	
		# Process key
		if key == ESC_KEY:
			running = False
		
		# Decide
		if line_pos is not None: 
			rotate = (line_pos - 0.5) * 0.2
		
		# Action 
		car.move(0.2, rotate)
		
	except:
		
		running = False
		
cv2.destroyWindow(frame_name)			
	

