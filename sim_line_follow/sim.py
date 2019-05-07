#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from math import cos, sin

# Keys
ESC_KEY = 27

# Point = np.array.shape [2,]

# Line class (p1, p2 should be np.matrix objects)		
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
		nom = np.cross(CA,AB) 
		denom = np.cross(CD, AB) ### Just check length AB and CD?
		if denom != 0:
			s = nom / denom
			i  = other.points[0,:] + s * CD
			overlap = self.in_range(i) and other.in_range(i)
			return(i, True, overlap)
		else:
			return None, False, False
		
	def in_range(self, p):
		return(((self.points[0,:] <= p)==(p <= self.points[1,:])).all())
			
# Function to plot line consitent of two dimensional array
def plot_line(M):
	for i in range((M.shape[0] - 1)):
		plt.plot([M[i, 0], M[i + 1, 0]],[M[i, 1], M[i + 1, 1]],'k-')
	plt.show()		
	
class Spline(object):
	
	def rel_line(self, S, E, ratio):
		return(S + (E - S) * ratio)
	
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

# Function to merge line segments
def merge_lines(line_list):
	line = np.concatenate(line_list)
	return(line)
		
# Course class
class Course(object):
	
	def __init__(self):				
		
		sect1 = Line(np.array([0,0]), np.array([1,0]))
		sect2 = Line(np.array([5,5]), np.array([4,6]))
		sect3 = Line(np.array([-4,6]), np.array([-5,5]))

		sp1 = Spline(sect1, sect2, 25) ### Make lean
		sp2 = Spline(sect2, sect3, 25)
		sp3 = Spline(sect3, sect1, 25)

		line_list = [sect1.points,
			   sp1.points,
			   sect2.points,
			   sp2.points,
			   sect3.points,
			   sp3.points]
		self.points = merge_lines(line_list)
	
	def draw(self, frame):
		for i in range(self.points.shape[0] - 1):		
			s_pix = to_pixel(self.points[i+0,:])
			e_pix = to_pixel(self.points[i+1,:])		
			cv2.line(frame, tuple(s_pix.astype(int)), tuple(e_pix.astype(int)),(255,0,0),2)		
		return frame

def rotation(rad):
	return(np.matrix([[cos(rad), -sin(rad)],
					[sin(rad), cos(rad)]]))	

# Car class
class Car(object):
	
	def __init__(self):
		self.pos = np.array([[0],[0]])		
		self.dir = np.array([[1],[0]])
		
	def move(self, x, rad):
		self.pos = self.pos + x * self.dir * 0.5
		R = np.matrix([[cos(rad), -sin(rad)],[sin(rad), cos(rad)]]) # Use rotation method
		self.dir = np.matmul(R, self.dir)
		self.pos = self.pos + x * self.dir * 0.5
	
	def draw(self, frame):
		sp = np.squeeze(np.array(self.pos))		
		ep = np.squeeze(np.array(self.pos - self.dir))		
		s_pix = to_pixel(sp)
		e_pix = to_pixel(ep)	
		cv2.line(frame, tuple(s_pix.astype(int)), tuple(e_pix.astype(int)),(0,0,255),2)		
		return frame
	
	def detect(self, points, frame):
		
		# Create detection line
		sp = self.pos + np.matmul(rotation(math.pi * -0.25), self.dir)
		ep = self.pos + np.matmul(rotation(math.pi *  0.25), self.dir)
		sp = np.squeeze(np.array(sp)) ### Fix the need to this transformation
		ep = np.squeeze(np.array(ep)) 
		detect_line = Line(sp, ep)
		
		# Draw detection line
		s_pix = to_pixel(np.squeeze(np.array(sp)))
		e_pix = to_pixel(np.squeeze(np.array(ep)))
		cv2.line(frame, tuple(s_pix.astype(int)), tuple(e_pix.astype(int)),(0,255,0),2)	
				
		# Loop through course
		for i in range(points.shape[0] - 1):
			sub_line = Line(points[i,:], points[i+1,:])
			I = detect_line.intersect(sub_line)			
			inters, does_intersect, overlap = I
			if overlap == True: 	
				return(np.linalg.norm(inters - sp) / np.linalg.norm(ep - sp))
	
# Function to convert cartesian coordinate into pixel coordinates
def to_pixel(cart):	 ### Move to class
	S = np.array([scale, -scale])
	T = np.array([320, 480 - 40])
	return(cart * S + T)

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
	rotate = (line_pos - 0.5) * 0.2
	print(rotate)
	
	# Action 
	car.move(0.15, rotate)
		
cv2.destroyWindow(frame_name)			
	

