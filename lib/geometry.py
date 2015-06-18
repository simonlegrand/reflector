""" Module containing geometric functions:
barycentre, distance from a point to a line
and furthest point from a given point
"""
from __future__ import print_function
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def barycentre(pts, w=None):
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
	
		if w is None:
			w = np.ones(len(pts))
			
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
	These expression are given for an incident light on Oz axis.
	
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
			
		diag = np.sqrt(eta*eta + ksi*ksi)
		r = np.sqrt(diag*diag + d*d)

		r_eta = np.sqrt(d*d + eta*eta)

		theta = np.arccos(d*np.cos(theta_0)/r_eta) - np.arcsin(ksi/r)

		if theta_0==0 and ksi==0:
			phi = phi_0 - np.pi/2
		else:
			phi = phi_0 - np.arctan(eta/(d*np.sin(theta_0)-ksi*np.cos(theta_0)))
		
		return r,theta,phi
		
	except FloatingPointError:
		print("****planar_to_spherical error: division by zero.")
		
	except NotProperShapeError, arg:
		print("****planar_to_spherical error: ", arg.msg)

def spherical_to_cartesian(r, theta, phi):
	"""
    This function transforms spherical coordinates r, theta, phi,
    into cartesian coordinates

    Parameters
    ----------
    r : real
    	Distance of the point from the origin
    theta : real
        Coordinate theta of the point
    phi : real
        Coordinate phi of the point

    Returns
    -------
    x, y, z : reals
        Cartesian coordinates of the point
    """
	try:
		r = np.asarray(r, dtype=np.float64)
		theta = np.asarray(theta, dtype=np.float64)
		phi = np.asarray(phi, dtype=np.float64)
		
		if len(r.shape) > 1 or len(theta.shape) > 1 or len(phi.shape) > 1:
			raise NotProperShapeError("r, theta, phi must be 1D arrays.")

		if r.shape != theta.shape or r.shape != phi.shape:
			raise NotProperShapeError("ksi and eta must have the same length.")

		x = r * np.sin(theta) * np.cos(phi)
		y = r * np.sin(theta) * np.sin(phi)
		z = r * np.cos(theta)

		return x,y,z

	except NotProperShapeError, arg:
		print("****spherical_to_cartesian error: ", arg.msg)

def plan_cartesian_equation(theta_0, phi_0, d):
	"""
	This function computes the cartesian equation of the plane
	which center has (theta_0, phi_0, d) spherical coordinates
	and which normal vector is directed toward the origin.

	Parameters
	----------
	theta_0 : real
		Coordinate theta of the plane origin
	phi_0 : real
		Coordinate phi of the plane origin
	d : real
		Distance from the plane origin to the origin

	Returns
	-------
	a, b, c, d : reals
		Coefficients of the cartesian equation
		ax + by + cz + d = 0 of the plan.
    """
    # Plan origin in cartesian coordinates
	plan_origin_sph = [d,theta_0,phi_0]
	plan_origin_cart = np.asarray(spherical_to_cartesian(plan_origin_sph[0],plan_origin_sph[1],plan_origin_sph[2]))

	eta_sph = planar_to_spherical(1.,0.,theta_0,phi_0,d)
	eta_cart = np.asarray(spherical_to_cartesian(eta_sph[0],eta_sph[1],eta_sph[2]))
	
	ksi_sph = planar_to_spherical(0.,1.,theta_0,phi_0,d)
	ksi_cart = np.asarray(spherical_to_cartesian(ksi_sph[0],ksi_sph[1],ksi_sph[2]))

	# Direction vectors of the plan in cartesian coordinates
	u = eta_cart - plan_origin_cart
	v = ksi_cart - plan_origin_cart

	# Coefficients of the plan cartesian equation
	a = u[1] * v[2] - u[2] * v[1]
	b = u[2] * v[0] - u[0] * v[2]
	c = u[0] * v[1] - u[1] * v[0]
	d = -(plan_origin_cart[0] * a + plan_origin_cart[1] * b + plan_origin_cart[2] * c) 
	
	return a,b,c,d
	
def plot_plan(a, b, c, d):
	# plane equation is a*x+b*y+c*z+d=0
	# create support in (x0y) plane
	xx, yy = np.meshgrid(range(-5,5), range(-5,5))

	# calculate corresponding z
	z = (-a * xx - b * yy - d) * 1. /c

	# plot the surface
	plt3d = plt.figure().gca(projection='3d')
	plt3d.plot_surface(xx, yy, z)
	plt.show()

def planar_to_gradient(eta, ksi, base, s1=None):
	"""
	This function computes the surface derivatives of the reflector
	for incident rays s1 and impact points of reflected rays in (eta,ksi)
	Parameters
	----------
	eta : 1D array
		Coordinate eta on the target plane
	ksi : 1D array
		Coordinate ksi on the target plane
	base : [0]e_eta , [1]e_ksi : Direct orthonormal 
		   basis of the target plan
		   [2]n_plan : Normal vector to the target plan
		   Its norm equals distance from plan to reflector.
	s1 : (1,3) array
		Incident ray direction

	Returns
	-------
	p,q : 1D arrays
		surface derivatives of the reflector
		
	See Also
	--------
	Inverse Methods for Illumination Optics, Corien Prins, chapter 5.3.1
    """
	e_eta = base[0]
	e_ksi =base[1]
	n = base[2]
	
	if s1 is None:
		s1 = np.array([0.,0.,1.])
	else:
		s1 = s1 / np.linalg.norm(s1)
	try:
		# Distance target plan/reflector
		d = np.linalg.norm(n)
		if d==0:
			raise ZeroDivisionError
		n = n / d
	
		# Reflected rays
		# The reflector is considered ponctual and
		# as the origin of the coordinate system
		s2 = np.zeros((len(eta),3))
		s2[:,0] = eta*e_eta[0] + ksi*e_ksi[0] - d*n[0]
		s2[:,1] = eta*e_eta[1] + ksi*e_ksi[1] - d*n[1]
		s2[:,2] = eta*e_eta[2] + ksi*e_ksi[2] - d*n[2]
	
		s2 = s2 / np.linalg.norm(s2, axis=1)[:, np.newaxis]
	
		p = -(s2[:,0] - s1[0])/(s2[:,2] - s1[2])
		q = -(s2[:,1] - s1[1])/(s2[:,2] - s1[2])
	
		return p, q
		
	except ZeroDivisionError:
		print("****planar_to_gradient error")

	
def rotation(pts, param):
	"""
	Rotate the reflector so as to place
	the concave face upwards.
	"""
	s1 = np.array([0,0,1])
	n = param['n_plan']/np.linalg.norm(param['n_plan'])
	
	axis = param['e_eta']
	angle = np.pi - np.arccos(np.dot(s1,n))/2

	I = np.identity(3)
	Q = np.matrix([[0,-axis[2],axis[1]],
					[axis[2],0,-axis[0]],
					[-axis[1],axis[0],0]])
	R = I + np.sin(angle)*Q + (1-np.cos(angle))*Q.dot(Q)
	for i in range(len(pts)):
		pts[i] = np.dot(R,pts[i])
	return pts
