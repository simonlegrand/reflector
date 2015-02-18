""" Module containing functions for the ray tracer"""
import numpy as np

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

def reflection(grad, s1=None):
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
	print(s1)
	print(grad)
	if s1 is None:
		s1 = np.array([0.,0.,1.])
	try:		
		N = s1.shape[0]

		if grad.shape[0] != N:
			raise NotProperShapeError("All arguments must have the same length.")	
		if grad.shape != (N,2):
			raise NotProperShapeError("grad must be (N,2).")
			
		nz = np.ones(N)
		n = np.array([grad[:,0],grad[:,1],-nz]).T
		n = n / np.linalg.norm(n, axis=-1)[:, np.newaxis]
		inner_product = s1[:,0] * n[:,0] + s1[:,1] * n[:,1] + s1[:,2] * n[:,2]
		s2 = s1 - 2 * inner_product[:, np.newaxis] * n
		
		return s2
	
	except NotProperShapeError, arg:
		print("****reflection error: ", arg.msg)


def ray_tracer(s1, interpol, e_eta, e_ksi, n_plan):

	##### Creation d'une grille X grace a laquelle on va evaluer le gradient de interpol
	nmesh = 30*N;
	[x,y] = np.meshgrid(np.linspace(0.,1.,nmesh),np.linspace(0.,1.,nmesh))

	Nx = nmesh*nmesh;

	x = np.reshape(x,(Nx))
	y = np.reshape(y,(Nx))
	X = np.vstack([x,y]).T
	I=np.reshape(interpol(X),(nmesh,nmesh))

	[gx, gy] = np.gradient(I, 1./nmesh, 1./nmesh)

	gx = np.reshape(gx, (nmesh*nmesh))
	gy = np.reshape(gy, (nmesh*nmesh))	
