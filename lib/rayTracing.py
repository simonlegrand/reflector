""" Module containing functions for the ray tracer"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sparse

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


def ray_tracer(s1, s_box, t_box, interpol, e_eta, e_ksi, n_plan, niter=None):
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
	e_eta , e_ksi : Direct orthonormal 
		basis of the target plan
	n_plan : Normal vector to the target plan
	"""	
	
	plot_reflector(interpol, s_box)

	M = None
	if niter is None:
		niter = 10
	for i in xrange(niter):
		nray = 500000
		x = s_box[0] + np.random.rand(nray) * (s_box[1] - s_box[0])
		y = s_box[2] + np.random.rand(nray) * (s_box[3] - s_box[2])
		grid = np.vstack([x,y]).T
		s2 = reflection(grid, interpol, s1)

		##### New spherical coordinates #####
		# psi is the inclination with respect
		# to -n_plan and khi is the azimutal 
		# angle around n_plan
		d = np.linalg.norm(n_plan)
		psi = np.arccos(np.inner(-s2,n_plan/d))

		a = np.inner(s2,e_ksi)
		b = np.inner(s2,e_eta)

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
		ksi = d * np.tan(psi) * np.cos(khi)
		eta = d * np.tan(psi) * np.sin(khi)

		eta_min = np.ones(len(eta)) * t_box[0]
		eta_max = np.ones(len(eta)) * t_box[1]
		ksi_min = np.ones(len(eta)) * t_box[2]
		ksi_max = np.ones(len(eta)) * t_box[3]
		J = np.logical_and(np.less_equal(eta,eta_max),
					       np.greater_equal(eta,eta_min))
		K = np.logical_and(np.less_equal(ksi,ksi_max),
					       np.greater_equal(ksi,ksi_min))
		L = np.logical_and(J,K)

		ksi = ksi[L]
		eta = eta[L]
		Miter = fill_sparse_matrix(eta, ksi)
		if M is None:
			M = Miter
		else:
			m = max(M.shape[0], Miter.shape[0])
			n = max(M.shape[1], Miter.shape[1])
			R = np.zeros((m,n))
			R[:Miter.shape[0], :Miter.shape[1]] = Miter
			R[:M.shape[0], :M.shape[1]] = R[:M.shape[0], :M.shape[1]] + M
			M = R
	
		print((i+1)*500000," rays thrown")
	M = 255.0*M/np.amax(M)
		    
	plt.imshow(M, interpolation='nearest',
		       vmin=0, vmax=255, cmap=plt.get_cmap('gray'));
	plt.show()
	
def plot_reflector(I, box):
	"""
	This function plots the interpolant I of the 
	functionnal evaluated on a grid
	"""
	nmesh = 100;
	[x,y] = np.meshgrid(np.linspace(box[0], box[1], nmesh),np.linspace(box[2], box[3], nmesh))
	Nx = nmesh*nmesh;
	x = np.reshape(x,(Nx))
	y = np.reshape(y,(Nx))
	grid = np.vstack([x,y]).T
	
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(grid[:,0],grid[:,1],I(grid[:,0],grid[:,1]), marker=".", lw=0.1)
	plt.show()
	
def fill_sparse_matrix(x,y):

	# Size conservation
	w = np.max(x) - np.min(x)
	h = np.max(y) - np.min(y)
	ratio = h / w
	n_linepix = 512
	n_columnpix = int(n_linepix / ratio)
	
	nmesh = len(x)
	j = np.floor((x-np.min(x))*n_columnpix/w)
	i = np.floor((np.max(y)-y)*n_linepix/h)
	data = np.ones(nmesh)
	
	M = sparse.coo_matrix((data, (i,j))).todense()
	Mmax = np.max(M)
	M = M * 255/Mmax
	return M
