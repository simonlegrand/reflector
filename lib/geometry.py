"""Module containing geometric functions:
barycentre, distance from a point to a line
and furthest point from a given point
"""
from __future__ import print_function
import sys
import numpy as np

class GeometricError(Exception):
	"""
	Base class for Geometric exceptions
	"""
	def __init__(self, arg):
		# Set some exception infomation
		self.msg = arg

class NotProperShapeError(GeometricError):
	"""Raised when the inputs have not the proper shape"""
	pass

def barycentre(pts, w):
	"""
	Compute the barycentre of the points cloud pts
	affected with w weights
	"""
	try:
		pts = np.asarray(pts, dtype=np.float64)
		w = np.asarray(w, dtype=np.float64)

		if pts.shape[0] != w.shape[0]:
			raise NotProperShapeError("Pts and w must have the same length.")
				
		if pts.shape[0] == 1:
			print ("barycentre warning: Only one point.")
			return pts
		
		bary = np.zeros(pts.shape[1])
		for i in range(pts.shape[1]):
			bary[i] = np.sum(w * pts[:,i])/np.sum(w)
		return bary
			
	except NotProperShapeError, arg:
		print("****Error: ", arg.msg)
		
def distPtLine(m, n, pt):
	"""
	Determine the distance between the line generated 
	by segment MN and the point pt
	"""
	try:
		m = np.asarray(m, dtype=np.float64)
		n = np.asarray(n, dtype=np.float64)
		pt = np.asarray(pt, dtype=np.float64)
	
		if len(m.shape) > 1 or len(n.shape) > 1 or len(pt.shape) > 1:
			raise NotProperShapeError("m, n and pt must be points (1D array).")
			
		if m.shape != n.shape or n.shape != pt.shape:
			raise NotProperShapeError("m, n and pt must have the same dimension.")
			
		if np.allclose(m, n):
			raise ValueError

		u = n - m			# Orientation vector
		Mpt = pt - m
		normU = np.linalg.norm(u)
		dist = np.linalg.norm(Mpt - (np.inner(Mpt,u)/(normU*normU))*u)
		return dist
	
	except NotProperShapeError, arg:
		print("****Error: ", arg.msg)
	except ValueError:
		print ("****Error: Impossible to generate a line from two identical points")


def furthestPt(cloud, a):
	"""
	Returns the distance between a and the furthest point in cloud
	Args:
		cloud is a (n, d) array, with n the nb of points, 
		d the dimension, a is a (,d) array
	"""
	try:
		a = np.asarray(a, dtype=np.float64)
		cloud = np.asarray(cloud, dtype=np.float64)
		
		if len(a.shape) > 1:
			raise NotProperShapeError("a must be a point")
			
		if a.shape[0] != cloud.shape[1]:
			raise NotProperShapeError("a and cloud must have the same number of colums.")
			
		if cloud.shape[0] == 1:
			print ("furthestPt warning: Only one point to compare")
			return cloud

		nbPts = np.shape(cloud)[0]
		dim = np.shape(cloud)[1]
		tmp = np.ones((nbPts, dim))		# Avoids the use of a for loop over nbPts
		for i in range(dim):
			tmp[:,i] = tmp[:,i]*a[i]
		dist = np.linalg.norm(tmp-cloud, axis=1)
		return np.max(dist)
		
	except NotProperShapeError, arg:
		print("****Error: ", arg.msg)
		
	except NotProperShapeError, arg:
		print("****Error: ", arg.msg)
