from __future__ import print_function
import sys
sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')

import numpy as np
import scipy as sp
import cPickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import MongeAmpere as ma

def init_parameters(parser):
	"""
	Initialise parameters of the problem according
	to a parameter file or by default if none
	is provided.
	
	Parameters
	----------
	parser : parser object
		Contains the parameter file name.
		
	Returns
	-------
	param : dictionnary
		Contains source and target files name,
		target base coordinates and source and
		target dimension.
	"""
	parser.add_argument('--f', '--file', type=str, default='0', help='parameter file', metavar='f')
	args = parser.parse_args()
	
	param = {}
	#### Default source and target ####
	param['source'] = 'default'
	param['target'] = 'default'
	#### Default target plan base ####
	param['e_eta'] = [0.,-1,0.]
	param['e_ksi'] = [0.,0.,1.]
	param['n_plan'] = [5.,0.,0.]
	#### Default source and target size ####
	# Useful only for pictures, because points
	# coordinates are already set by the input
	# file if any is given.
	# The size is given by the largest dimension
	# of the picture, proportions are then conserved.
	param['s_size'] = 1.
	param['t_size'] = 10.
	
	# If a parameter file is given
	if args.f != '0':
		infile = None
		try:
			infile = open(args.f, "r")
			
			try:
				header = infile.readline().rstrip('\n\r')
		
				if header != 'Reflector parameter file':
					raise ValueError
				
				else:
					# Parameters contained in the file
					# will erase the default ones. The
					# others will remain by default.
					for line in infile:
						data = line.rstrip('\n\r').split("\t")
						if data[0] in param:
							param[data[0]] = data[1:]
				
					e_eta = np.array([float(x) for x in param['e_eta']])
					e_ksi = np.array([float(x) for x in param['e_ksi']])
					n_plan = np.array([float(x) for x in param['n_plan']])
					param['s_size'] = float(param['s_size'][0])
					param['t_size'] = float(param['t_size'][0])
					param['source'] = str(param['source'][0])
					param['target'] = str(param['target'][0])
				
					# Assert the base is orthogonal and 
					# normalize e_eta and e_ksi
					assert(np.dot(e_eta,e_ksi)==0.)
					cross_prod = np.cross(np.cross(e_eta,e_ksi),n_plan)
					assert(np.allclose(cross_prod,np.zeros(3)))
					e_eta /= np.linalg.norm(e_eta)
					e_ksi /= np.linalg.norm(e_ksi)
					param['e_eta'] = e_eta
					param['e_ksi'] = e_ksi
					param['n_plan'] = n_plan

			finally:
				infile.close()
				
		except IOError:
			print ("Error: can\'t find file or read data")
			sys.exit()
			
		except ValueError:
			print ("Error: wrong data type in the file")
			sys.exit()
	
	return param
	
def display_parameters(param):
	for keys in param:
		print(keys,' : ', param[keys])
		
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

def write_data(x, y, u, grad, filename):
	"""
	Write a function and its gradient
	evaluated on a grid in a file
	"""
	try:
		outfile = open(filename,'w')
		data = np.array([x,y,u,grad[:,0],grad[:,1]])
		print(np.shape(data))
		for i in range (0, len(data)):
			outfile.write(data[i,:])
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

def save_reflector(I, box, filename):
	"""
	Write a function and its gradient
	evaluated on a grid in a file
	"""
	nmesh = 400
	[x,y] = np.meshgrid(np.linspace(box[0], box[1], nmesh),np.linspace(box[2], box[3], nmesh))
	Nx = nmesh*nmesh
	x = np.reshape(x,(Nx))
	y = np.reshape(y,(Nx))
	grid = np.vstack([x,y]).T
	
	u = I(x,y)
	grad = I.gradient(x,y)
	J = np.logical_or(~u.mask, ~grad[0].mask, ~grad[1].mask)
	grid = grid[J]
	u = u[J]
	grad = [grad[0][J],grad[1][J]]

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
	# on trouve un point dans chaque cellule de Laguerre
	Z = mu.lloyd(Y,psi)[0]
	
	# par definition, psi^*(z) = min_{y\in Y} |y - z|^2 - psi_y
	# Comme Z[i] est dans la cellule de Laguerre de Y[i], la formule se simplifie:
	psi_star = np.square(Y[:,0] - Z[:,0]) + np.square(Y[:,1] - Z[:,1]) - psi
	T = ma.delaunay_2(Z, psi_star)
	
	# ensuite, on modifie pour retrouver une fonction convexe \tilde{psi^*} telle
	# que \grad \tilde{\psi^*}(z) = y si z \in Lag_Y^\psi(y)
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
