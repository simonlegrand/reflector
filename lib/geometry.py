"""Module containing geometric functions:
barycentre, distance from a point to a line
and furthest point from a given point
"""
from __future__ import print_function
import sys

import numpy as np

# To raise an error instead of a RuntimeWarning
# when there is a division by zero
np.seterr(divide='raise')

class GeometricError(Exception):
	"""
	Base class for Geometric exceptions
	"""
	def __init__(self, arg):
		# Set some exception infomation
		self.msg = arg

class NotProperShapeError(GeometricError):
	"""Raised when inputs have not the proper shape"""
	pass

def barycentre(pts, w):
	"""
	Computes the barycentre of points cloud pts
	affected with w weights, in a space of arbitrary
	dimension dim.
	
	Parameters
	----------
	pts : array (N,dim)
		Cloud of points
	w : 1D array (N,)
		Weights associated to pts
		
	Returns
	-------
	bary : array (dim,)
		Coordinates of barycentre
	"""
	try:
		pts = np.asarray(pts, dtype=np.float64)
		w = np.asarray(w, dtype=np.float64)

		if pts.shape[0] != w.shape[0]:
			raise NotProperShapeError("Pts and w must have the same length.")
				
		if pts.shape[0] == 1:
			print ("barycentre warning: Only one point.")
			return pts
		
		dim = pts.shape[1]
		bary = np.zeros(dim)
		for i in range(dim):
			bary[i] = np.sum(w * pts[:,i])/np.sum(w)
		return bary
			
	except NotProperShapeError, arg:
		print("****barycentre error: ", arg.msg)
		
def distance_point_line(m, n, pt):
	"""
	Computes the distance between the line generated 
	by segment MN and the point pt, in a space of arbitrary
	dimension dim.
	
	Parameters
	----------
	m : array (dim,)
		Point generating the line
	n : array (dim,)
		Second point generating the line
	pt : array (dim,)
		Point we calculate the distance from the line
		
	Returns
	-------
	dist : real
		distance between line (MN) and pt
	
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

		u = n - m			# Direction vector
		Mpt = pt - m
		norm_u = np.linalg.norm(u)
		dist = np.linalg.norm(Mpt - (np.inner(Mpt,u)/(norm_u*norm_u))*u)
		return dist
	
	except NotProperShapeError, arg:
		print("****distance_point_line error: ", arg.msg)
		
	except ValueError:
		print ("****distance_point_line error: Impossible to generate a line from two identical points")


def furthest_point(cloud, a):
	"""
	Computes the distance between a and the furthest point
	in cloud, in a space of arbitrary dimension dim.
	
	Parameters
	----------
	cloud : (N,dim) array
		Point cloud
	a : (,dim) array
		Point
	
	Returns
	-------
	distance : real
		distance between a and the furthest point in cloud.
	"""
	try:
		a = np.asarray(a, dtype=np.float64)
		cloud = np.asarray(cloud, dtype=np.float64)
		
		if len(a.shape) > 1:
			raise NotProperShapeError("a must be a point")
			
		if a.shape[0] != cloud.shape[1]:
			raise NotProperShapeError("a and cloud must have the same number of colums.")
			
		if cloud.shape[0] == 1:
			print ("furthest_point warning: Only one point to compare")
			return cloud

		N = np.shape(cloud)[0]
		dim = np.shape(cloud)[1]
		tmp = np.ones((N, dim))	# Avoids the use of a for loop over nbPts
		for i in range(dim):
			tmp[:,i] = tmp[:,i]*a[i]
		dist = np.linalg.norm(tmp-cloud, axis=1)
		return np.max(dist)
		
	except NotProperShapeError, arg:
		print("****furthest_point error: ", arg.msg)
		

def gradient_to_spherical(gx,gy):
	"""
	This function convert gradient coordinates of the 
	reflector into spherical coordinates of reflected rays
	on the unit sphere S2.
	
	Parameters
	----------
	gx : 1D array
		Gradients coordinate along x axis
	gy : 1D array
		Gradients coordinate along y axis
	
	Returns
	-------
	theta : 1D array
		Inclination angles (with respect to the
		positiv z axis). 0 <= theta <= pi
	phi : 1D array
		Azimuthal angles (projection of a direction
		in z=0 plane with respect to the x axis).
		0 <= phi <= 2pi
		
	See Also
	--------
	Inverse Methods for Illumination Optics, Corien Prins
	"""
	try:
		gx = np.asarray(gx, dtype=np.float64)
		gy = np.asarray(gy, dtype=np.float64)
		
		if len(gx.shape) > 1 or len(gy.shape) > 1:
			raise NotProperShapeError("gx and gy must be 1D arrays.")
		
		if gx.shape != gy.shape:
			raise NotProperShapeError("gx and gy must have the same length.")
			
		# theta computation
		num = gx*gx + gy*gy - 1
		denom = gx*gx + gy*gy + 1
		theta = np.arccos(num/denom)

		# phi computation
		zero = np.zeros(gx.shape)
		phi = np.zeros(gx.shape)
		J = np.logical_and(np.greater_equal(gx,zero),np.greater_equal(gy,zero))
		phi[J] = np.arctan(gy[J]/gx[J])
			
		J = np.less(gx, zero)
		phi[J] = np.arctan(gy[J]/gx[J]) + np.pi
			
		J = np.logical_and(np.greater_equal(gx, zero), np.less(gy, zero))
		phi[J] = np.arctan(gy[J]/gx[J]) + 2*np.pi
			
		return theta, phi
		
	except FloatingPointError:
		print("****gradient_to_spherical error: division by zero.")
		
	except NotProperShapeError, arg:
		print("****gradient_to_spherical error: ", arg.msg)
		
def spherical_to_gradient(theta,phi):
	"""
	This function convert spherical coordinates of reflected rays
	on the unit sphere S2 into gradient coordinates of the reflector.
	
	Parameters
	----------
	theta : 1D array
		Inclination angles (with respect to the
		positiv z axis). 0 <= theta <= pi
	phi : 1D array
		Azimuthal angles (projection of a direction
		in z=0 plane with respect to the x axis).
		0 <= phi <= 2pi
	
	Returns
	-------
	gx : 1D array
		Gradients coordinate along x axis
	gy : 1D array
		Gradients coordinate along y axis
		
	See Also
	--------
	Inverse Methods for Illumination Optics, Corien Prins
	"""
	try:
		theta = np.asarray(theta, dtype=np.float64)
		phi = np.asarray(phi, dtype=np.float64)
		
		if len(theta.shape) > 1 or len(phi.shape) > 1:
			raise NotProperShapeError("theta and phi must be 1D arrays.")
		
		if theta.shape != phi.shape:
			raise NotProperShapeError("theta and phi must have the same length.")
		
		gx = np.sin(theta) * np.cos(phi) / (1 - np.cos(theta))
		gy = np.sin(theta) * np.sin(phi) / (1 - np.cos(theta))
		
		return gx,gy
		
	except FloatingPointError:
		print("****spherical_to_gradient error: division by zero.")
		
	except NotProperShapeError, arg:
		print("****spherical_to_gradient error: ", arg.msg)
		
def planar_to_spherical(eta,ksi,theta_0,phi_0,d):
	"""
	This function computes to conversion from planar coordinates
	(eta, ksi) to spherical coordinates (theta, phi) on the unit sphere
	for the reflector problem.
	
	Parameters
	----------
	eta : 1D array
		Forms an orthonormal basis with ksi and the 
		normal vector of the plane directed to 
		the center of unit sphere
	ksi : 1D array
		Coordinate along ksi axis parallel to z axis
	theta_0 : real
		Coordinate theta of the plane origin
	phi_0 : real
		Coordinate phi of the plane origin	
	d : Distance from the plane origin to the center
		of unit sphere
		
	Returns
	-------
	theta : 1D array
		Inclination angles (with respect to the
		positiv z axis). 0 <= theta <= pi
	phi : 1D array
		Azimuthal angles (projection of a direction
		in z=0 plane with respect to the x axis).
		-pi/2 <= phi <= pi/2
		
	See Also
	--------
	Inverse Methods for Illumination Optics, Corien Prins
	"""
	try:
		eta = np.asarray(eta, dtype=np.float64)
		ksi = np.asarray(ksi, dtype=np.float64)
		
		if len(ksi.shape) > 1 or len(eta.shape) > 1:
			raise NotProperShapeError("ksi and eta must be 1D arrays.")
		
		if ksi.shape != eta.shape:
			raise NotProperShapeError("ksi and eta must have the same length.")
			
		phi = phi_0 - np.arctan(eta/d)
		theta = theta_0 - np.arctan(ksi*np.cos(phi_0-phi)/d)
		
		return theta,phi
			
	except FloatingPointError:
		print("****planar_to_spherical error: division by zero.")
		
	except NotProperShapeError, arg:
		print("****planar_to_spherical error: ", arg.msg)
