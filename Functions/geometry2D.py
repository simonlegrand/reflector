"""Module containing 2d geometric functions"""
from __future__ import print_function
import sys
import numpy as np
	
def dotProduct(a, b):
	"""Compute the dot product of a and b"""
	if np.shape(a) != np.shape(b):
		print ("dotProdut error: Arrays with different shapes")
		sys.exit()
		
	if len(np.shape(a)) == 1:
		return a[0] * b[0] + a[1] * b[1]
		
	elif len(np.shape(a)) > 1:
		return a[:,0] * b[:,0] + a[:,1] * b[:,1]

# dotProduct test
if __name__ == "__main__":
	u = np.array([[1.,0.],[0.,1.],[1.,0.],[2.,1.]])
	v = np.array([[2.,0.],[1.,1.],[1.,1.],[2.,1.]])
	print ("u = ", u)
	print ("v = ", v)
	print ("u.v = ", dotProduct(u,v))
	u = np.array([1.,0.])
	v = np.array([2.,0.])
	print ("u.v = ", dotProduct(u,v))
	
def dist(a, b):
	"""Compute the distance between points a and b"""
	if np.shape(a) != np.shape(b):
		print ("dist error: Arrays with different shapes")
		sys.exit()
	
	if len(np.shape(a)) == 1:
		return np.sqrt(np.power(b[0]-a[0], 2) + np.power(b[1]-a[1], 2))

	elif len(np.shape(a)) > 1:
		return np.sqrt(np.power(b[:,0]-a[:,0], 2) + np.power(b[:,1]-a[:,1], 2))

# dist test
if __name__ == "__main__":
	u = np.array([[1.,0.],[0.,1.],[1.,0.],[2.,1.]])
	v = np.array([[2.,0.],[1.,1.],[1.,1.],[2.,1.]])
	print ("dist(u,v) = ", dist(u,v))
	u = np.array([0.,0.])
	v = np.array([1.,1.])
	print ("dist(u,v) = ", dist(u,v))
	
def barycentre(pts, w):
	"""Compute the barycentre of the points cloud pts
	affected with w weights"""
	if len(np.shape(pts)) == 1 and len(w) == 1:
		print ("barycentre warning: Only one point")
		return pts

	elif len(pts) != len(w):
		print ("barycentre error: Arrays have not the same length")
		sys.exit()
		
	else:
		bx = np.sum(w * pts[:,0])/np.sum(w)
		by = np.sum(w * pts[:,1])/np.sum(w)
		return np.array([bx, by])

# Test of barycentre
if __name__ == "__main__":
	points = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
	weights = np.array([1., 2., 1., 2.])
	print ("barycentre = ", barycentre(points, weights))
	points = np.array([0.,0.])
	weights = np.array([1.])
	print ("barycentre = ", barycentre(points, weights))

def distPtLine(m, n, pt):
	"""
	Determine the distance between the line generated 
	by segment MN and the point pt
	Line eq: -(yn - ym)*x + (xn - xm)*y - (xn - xm)*ym + (yn - ym)*xm = 0
	"""
	m = np.array(m)
	n = np.array(n)
	pt = np.array(pt)
	
	if np.allclose(m, n):
		print ("distPtLine error: Impossible to generate a line from two identical points")
		sys.exit()
		
	elif len(np.shape(pt)) != 1:
		print ("distPtLine error: pt must a 1D array")
		sys.exit()
	
	a = -(n[1] - m[1])
	b = (n[0] - m[0])	
	c = -(n[0] - m[0])*n[1] + (n[1] - m[1])*n[0]
	dist = abs(a*pt[0] + b*pt[1] + c)/np.sqrt(a*a + b*b)
	return dist

"""d1 = distPtLine([1., 0.], [1., 1.], [0.5, 0.5])
d2 = distPtLine([0.0,0.0], [1.0, 0.0], [0.0, 1.0])
d3 = distPtLine([0.0,0.0], [-1.0, 0.0], [1.0, 0.0])
print d1, d2, d3"""

def furthestPt(cloud, a):
	"""
	Returns the distance between a and the furthest point in cloud
	Args:
		cloud is a (n, 2) array, with n the nb of points
		a is a (, 2) array containing coord of pt
	"""
	if np.shape(cloud) == (1, 2):
		print ("furthestPt warning: Only one point to compare")
		return cloud
	
	else:
		nbPts = np.shape(cloud)[0]
		tmp = np.ones((nbPts, 2))		#Avoids the use of a for loop
		tmp[:,0] = tmp[:,0]*a[0]
		tmp[:,1] = tmp[:,1]*a[1]
		dist = np.sqrt(np.power(cloud[:,0] - tmp[:,0], 2) + np.power(cloud[:,1] - tmp[:,1], 2))
		return np.max(dist)
"""
c = np.array([[0.0, 1.0], [0.0, 0.5], [0.0, -2.0], [3.0, 3.0]])
p = np.array([0.0, 0.0])
print furthestPt(c, p)
c = np.array([[0.0, 1.0]])
p = np.array([0.0, 0.0])
print furthestPt(c, p)"""
