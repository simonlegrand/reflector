from __future__ import print_function
import sys
sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')

import numpy as np
import cPickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.spatial import ConvexHull

import geometry as geo
import MongeAmpere as ma
		
def write_data1(data, filename):
	"""
	Serialize data and write it in a file
	"""
	try:
		outfile = open(filename,'w')
		cPickle.dump(data, outfile,0)
	except (NameError, IOError) as e:
		print(e)
		sys.exit(1)

def load_data(filename):
	try:
		infile = open(filename,'r')
		data = cPickle.load(infile)
		return data
	except (NameError, IOError) as e:
		print(e)
		sys.exit(1)
		
def plot_reflector(I, box):
	"""
	This function plots the interpolant I of the 
	functionnal evaluated on a grid
	"""
	nmesh = 100
	[x,y] = np.meshgrid(np.linspace(box[0], box[1], nmesh),np.linspace(box[2], box[3], nmesh))
	Nx = nmesh*nmesh
	x = np.reshape(x,(Nx))
	y = np.reshape(y,(Nx))
	grid = np.vstack([x,y]).T
	
	u = I(x,y)
	J = ~u.mask
	u = u[J]
	grid = grid[J]
		
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(grid[:,0], grid[:,1], u, cmap=cm.jet, linewidth=0.2)
	plt.show()


def plot_2d_points(x,y):
	fig = plt.figure()
	ax = plt.scatter(x,y)
	plt.show()
	
	
def save_reflector(I, box, filename):
	"""
	Write a function and its gradient
	evaluated on a grid in a file
	"""
	nmesh = 100
	[x,y] = np.meshgrid(np.linspace(box[0]+2e-2, box[1]-2e-2, nmesh),np.linspace(box[2]+2e-2, box[3]-2e-2, nmesh))
	Nx = nmesh*nmesh
	x = np.reshape(x,(Nx))
	y = np.reshape(y,(Nx))
	grid = np.vstack([x,y]).T
	print(np.shape(grid))
	
	u = I(x,y)
	grad = I.gradient(x,y)
	"""J = np.logical_or(~u.mask, ~grad[0].mask, ~grad[1].mask)
	grid = grid[J]
	u = u[J]""
	grad = [grad[0][J],grad[1][J]]"""
	print(np.shape(u))
	try:
		outfile = open(filename,'w')
		outfile.write('x' + '\t' + 'y' + '\t' + 'u' + '\t' + 'du/dx' + '\t' + 'du/dy' + '\n')
		for i in range (0, len(u)):
			outfile.write(str(grid[i,0]) + '\t' + str(grid[i,1]) + '\t' + str(u[i]) + '\t' + str(grad[0][i]) + '\t' + str(grad[1][i]) + '\n')
	except (NameError, IOError) as e:
		print(e)
		sys.exit(1)
		
def plot_density(mu):	
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	#ax.set_zlim3d(0,1.5)
	ax.plot_trisurf(mu.vertices[:,0], mu.vertices[:,1], mu.values, cmap=cm.jet, linewidth=0.2)
	plt.show()

def eval_legendre_fenchel(mu, Y, psi):
	"""
	This function returns centers of laguerre cells Z,
	the delaunay triangulation of these points and the 
	Legendre-Fenchel transform of psi.
	
	Parameters
	----------
	mu : Density_2 object
		Density of X ensemble
	Y : 2D array
		Points on the Y ensemble
	psi : 1D array
		Functionnal values on Y
			
	Returns
	-------
	Z : 2D array
		Laguerre cells centers coordinates
	T : delaunay_2 object
		Weighted Delaunay triangulation of Z
	psi_star_tilde : 1D array
		Legendre-Fenchel transform of psi
	"""
	# We find the center of each Laguerre cell
	# or their projection on the boundary for the cells that 
	# intersect the boundary
	X = mu.vertices
	hull = ConvexHull(X)
	points = X[hull.vertices]
	Z = ma.ma.conforming_lloyd_2(mu,Y,psi,points)[0]
	
	# By definition, psi^*(z) = min_{y\in Y} |y - z|^2 - psi_y
	# As Z[i] is in the Laguerre cell of Y[i], the formula can be simplified:
	psi_star = np.square(Y[:,0] - Z[:,0]) + np.square(Y[:,1] - Z[:,1]) - psi
	T = ma.delaunay_2(Z, psi_star)
	
	# Modification to get a convex function \tilde{psi^*} such
	# that \grad \tilde{\psi^*}(z) = y if z \in Lag_Y^\psi(y)
	psi_star_tilde = (np.square(Z[:,0]) + np.square(Z[:,1]))/2 - psi_star/2
	
	return Z,T,psi_star_tilde
	
def make_cubic_interpolator(Z,T,psi,grad):
	"""
	This function returns a cubic interpolant
	of function psi.
	"""
	T = tri.Triangulation(Z[:,0],Z[:,1],T)
	interpol = tri.CubicTriInterpolator(T, psi, kind='user', dz=(grad[:,0],grad[:,1]))
	return interpol
