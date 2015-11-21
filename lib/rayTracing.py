""" Module containing functions for the ray tracer"""
from __future__ import print_function
import sys
import numpy as np
import scipy.sparse as sparse
import numpy.matlib
sys.path.append('./lib')
import misc
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

def reflection(grid, I, s1):
	"""
	Computes directions s2 of reflected rays.
	Direct application of Snell & Descartes law.
	
	Parameters
	----------
	s1 : array (N,3)
		direction of incident rays
	x : array (N,)
	y : array (N,)
	f : array (N,)
		Reflector shape function
		
	Returns
	-------
	s2 : array (N,3)
		Direction of reflected rays.
	"""	
	[gx, gy] = I.gradient(grid[:,0], grid[:,1])
	J = np.logical_or(~gx.mask, ~gy.mask)
	gx = gx[J]
	gy = gy[J]
	
	nz = np.ones(gx.shape)
	n = np.vstack([gx,gy,-nz]).T
	n = n / np.linalg.norm(n, axis=-1)[:, np.newaxis]

	inner_product = s1[0] * n[:,0] + s1[1] * n[:,1] + s1[2] * n[:,2]
	s2 = s1 - 2 * inner_product[:, np.newaxis] * n

	return s2


def ray_tracer(density, t_box, interpol, base, niter=None, s1=[0.,0.,1.]):
	"""
	This function computes the simulation of reflection on the reflector
	and plot the image produced on the target screen.
	
	Parameters
	----------
	s1 : array (N,3)
		direction of incident rays
	s_box : tuple[4]
		enclosing square box of the source support
		[xmin, xmax, ymin, ymax]
	t_box : tuple[4]
		enclosing square box of the target support
		[ymin, xmax, ymin, ymax]
	interpol : TriCubic interpolant of the reflector
	base : [0]e_eta , [1]e_xi : Direct orthonormal 
		   basis of the target plan
		   [2]n_plan : Normal vector to the target plan
	"""	
	e_eta = base[0]
	e_xi = base[1]
	n_plan = base[2]
	
	M = None
	if niter is None:
		niter = 10
	for i in xrange(niter):
		nray = 200000
		# Generate source point according to
		# to the source density probability
		points = density.random_sampling(nray)
		s2 = reflection(points, interpol, s1)
		#print(min(s2[:,0]),max(s2[:,0]), min(s2[:,1]),max(s2[:,1]), min(s2[:,2]),max(s2[:,2]))
		##### New spherical coordinates #####
		# psi is the inclination with respect
		# to -n_plan and khi is the azimutal 
		# angle around n_plan
		d = np.linalg.norm(n_plan)
		n = n_plan/d
		
		scal = s2[:,0]*n[0] + s2[:,1]*n[1] + s2[:,2]*n[2]
		psi = np.arccos(-scal)
		
		a = s2[:,0]*e_xi[0] + s2[:,1]*e_xi[1] + s2[:,2]*e_xi[2]
		b = s2[:,0]*e_eta[0] + s2[:,1]*e_eta[1] + s2[:,2]*e_eta[2]
		
		khi = np.zeros(a.shape)
		zero = np.zeros(a.shape)
		J = np.logical_and(np.greater_equal(a,zero),
						   np.greater_equal(b,zero))
		khi[J] = np.arctan(b[J]/a[J])

		J = np.less(a, zero)
		khi[J] = np.arctan(b[J]/a[J]) + np.pi

		J = np.logical_and(np.greater_equal(a, zero), np.less(b, zero))
		khi[J] = np.arctan(b[J]/a[J]) + 2*np.pi

		##### Planar coordinates #####
		# computation of intersection of reflected rays
		# on the target plan and selection of points
		# inside t_box
		xi = d * np.tan(psi) * np.cos(khi)
		eta = d * np.tan(psi) * np.sin(khi)

		eta_min = np.ones(len(eta)) * t_box[0]
		eta_max = np.ones(len(eta)) * t_box[1]
		xi_min = np.ones(len(eta)) * t_box[2]
		xi_max = np.ones(len(eta)) * t_box[3]
		J = np.logical_and(np.less_equal(eta,eta_max),
						   np.greater_equal(eta,eta_min))
		K = np.logical_and(np.less_equal(xi,xi_max),
						   np.greater_equal(xi,xi_min))
		L = np.logical_and(J,K)

		xi = xi[L]
		eta = eta[L]

		Miter = fill_sparse_matrix(eta, xi, t_box)
		if M is None:
			M = Miter
		else:
			M += Miter
	
		print("it", i+1,":", (i+1)*nray,"rays thrown")
	M = 255.0*M/np.amax(M)
		    
	return M

def fill_sparse_matrix(x,y,box):

	# Size conservation
	w = box[1] - box[0]
	h = box[3] - box[2]
	ratio = h / w
	n_linepix = 256
	n_columnpix = int(n_linepix / ratio)
	
	nmesh = len(x)
	j = np.floor((x-np.min(x))*n_columnpix/w)
	i = np.floor((np.max(y)-y)*n_linepix/h)
	data = np.ones(nmesh)
	
	M = sparse.coo_matrix((data, (i,j)), shape=(n_linepix,n_columnpix)).todense()
	return M
