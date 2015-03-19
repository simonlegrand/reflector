import numpy as np
import scipy as sp
import cPickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

def write_data(data, filename):
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
	
	I = I(x,y)
	Z = I[~np.isnan(I)]
	grid = grid[~np.isnan(I)]
		
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot_trisurf(grid[:,0], grid[:,1], Z, cmap=cm.jet, linewidth=0.2)
	plt.show()
	
def plot_density(X, mu):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	#ax.set_zlim3d(0,1.5)
	ax.plot_trisurf(X[:,0], X[:,1], mu, cmap=cm.jet, linewidth=0.2)
	plt.show()
	
