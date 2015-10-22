"""Contains functions used during the input processing step."""
from __future__ import print_function
import sys
sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy as sp
from scipy import ndimage

import MongeAmpere as ma
import misc
import geometry as geo

def rgb_to_gray(imcolor):
	"""
	Conversion of a rgb picture into a grayscaled picture.
	
	Parameters
	----------
	imcolor : 3D array
		Three layers RGB.
		
	Returns
	-------
	imgray : 2D array
		Grayscaled picture
	"""
	imgray = 0.299*imcolor[:,:,0] + 0.587*imcolor[:,:,1] + 0.114*imcolor[:,:,2]
	return imgray
	
def input_preprocessing(parameters):
	"""
	Generates proper inputs according to
	parameters. A density for the source
	and a diracs sum for the target.
	
	Parameters
	----------
	parameters : dictionnary
		Contains all the parameters
		
	Returns
	-------
	mu : Density_2 object
	Y : 2D array
	nu : 1D array
	"""
	source_name = parameters['source'][0]
	source_geom = parameters['source'][1:]
	
	target_name = parameters['target'][0]
	target_geom = parameters['target'][1:]

	#### Source processing ####
	if source_name == 'default':
		x = np.array([-0.5,-0.5,0.5,0.5])
		y = np.array([-0.5,0.5,-0.5,0.5])
		X = np.array([x,y]).T
		mu = ma.Density_2(X)
		
	else:
		extension_source = os.path.splitext(source_name)[1]
		
		if extension_source == ".txt":
			X, X_w = read_data(source_name, source_geom)
			mu = ma.Density_2(X, X_w)
		
		else:
			X, mu = read_image(source_name, source_geom)
			# Null pixels are removed
			zero = np.zeros(len(mu))
			J = np.greater(mu,zero)
			mu = mu[J]
			X = X[J]
			mu = ma.Density_2(X,mu)
		
	#### Target processing ####
	if target_name == 'default':
		# Number of diracs for the default shape
		Ndiracs = 10000
		# Triangle
		x = np.array([0.5,-0.25,-0.25])
		y = np.array([0.,0.433,-0.433])
		# Square
		#x = np.array([-0.5,-0.5,0.5,0.5])
		#y = np.array([-0.5,0.5,-0.5,0.5])
		#Pentagon
		#x = np.array([0.,-1.,-0.75, 0.75, 1.])
		#y = np.array([1.5,0.25,-0.5, -0.5, 0.25])
		tri = np.array([x,y]).T
		dens_target = ma.Density_2(tri)
		Y = ma.optimized_sampling_2(dens_target,Ndiracs)
		nu = np.ones(Ndiracs) * (mu.mass() / Ndiracs)
		
	else:
		extension_target = os.path.splitext(target_name)[1]

		if extension_target == ".txt":
			Y, nu = read_data(target_name, target_geom)
			nu = nu * (mu.mass() / sum(nu))
		
		else:
			# For a picture, number of dirac equals
			# number of pixels (set in read_image())
			Y, nu = read_image(target_name, target_geom)
			
			# Null pixels are removed
			zero = np.zeros(len(nu))
			J = np.greater(nu,zero)
			Y = Y[J]
			nu = nu[J]
			
			# Mass equalization
			nu = nu * (mu.mass()/sum(nu))

	return mu, Y, nu


def read_data(fn, geom):
	"""
	This function returns points coordinates
	contained in file fn, rescaled and repositionned
	according to geometric parameters.
	
	Parameters
	----------
	fn : string
		File adress
	geom : list	
		geometric parameters
	
	Returns
	-------
	X : 2D array
		Points coordinates of the polygon
	z : 1D array
		Weights associated to points X
	"""
	try:
		infile = open(fn, "r")
		if infile is None:
			raise IOError("Error: can\'t find", fn)
			
		try:
			header = infile.readline().rstrip('\n\r')
		
			if header != 'Reflector input file':
				raise ValueError("Error: wrong data type in", fn)
		
			x = []
			y = []
			z = []
			for line in infile:
				data = line.rstrip('\n\r').split("\t")
				x.append(float(data[0]))
				y.append(float(data[1]))
				if len(data) == 3:
					z.append(float(data[2]))
			
			x = np.asarray(x)
			y = np.asarray(y)
		
			if len(data) == 3:
				z = np.asarray(z)
			else:
				z = np.ones(len(x))
			
			X = np.array([x,y]).T
			
			# Rescaling and repositionning
			if len(geom)>0:
				bary = np.average(X,axis=0)
				X[:,0] = X[:,0] - bary[0]
				X[:,1] = X[:,1] - bary[1]
				ratio_x = (np.max(X[:,0])-np.min(X[:,0])) / geom[0]
				ratio_y = (np.max(X[:,1])-np.min(X[:,1])) / geom[1]
				if len(geom)==2:
					X[:,0] = X[:,0]/ratio_x
					X[:,1] = X[:,1]/ratio_y
				elif len(geom)==4:
					X[:,0] = X[:,0]/ratio_x + geom[2]
					X[:,1] = X[:,1]/ratio_y + geom[3]
				else:
					print("Error: geometric paramterers length must be 0,2 or 4")
						
			return X, z
				
		finally:
			infile.close()
			
	except (ValueError, IOError) as e:
		print(e)
		sys.exit(1)

def read_image(fn, geom):
	"""
	This function returns file fn pixels coordinates and value
	between 0 and 1. If the picture is in rgb, it is grayscaled.
	Coordinates are rescaled and repositionned according to
	geometric parameters.
	
	Parameters
	----------
	fn : string
		Picture adress
	geom : list	
		geometric parameters
		
	Returns
	-------
	X : 2D array
		Pixels coordinates of the picture
	img : 1D array
		Pixels value.
	"""
	try:
		if len(geom) == 0:
			raise ValueError("Error: width and height \
of a picture must be set in parameter file")

		image = sp.misc.imread(fn)

		dims = np.shape(image)
		if len(dims) == 3:
			img = rgb_to_gray(image)
		elif len(dims) != 2:
			raise ValueError("Error: Wrong dimension for the image")
		else:
			img = image

		# Rotation if needed
		#img = ndimage.rotate(img,-90)
		
		ratio = float(np.shape(img)[0]) / np.shape(img)[1]
		nlin = 200
		ncol = int(nlin / ratio)	
		img = sp.misc.imresize(img, (nlin,ncol))
		
		img = np.asarray(img, dtype=float)
		img = img / 255.0

		# Center the image
		xmin = -geom[0] / 2.
		ymin = -geom[1] / 2.
		box = [xmin,-xmin,-ymin,ymin]
		X = regular_sampling(img,box)
		# Repositionning if needed
		if len(geom)==4:
			X[:,0] += geom[2]
			X[:,1] += geom[3]
		
		img = np.reshape(img,(len(X)))
		return X , img
	
	except (ValueError, IOError) as e:
		print(e)
		sys.exit(1)

	
def regular_sampling(img,box):
	"""
	This function returns pixel cartesian
	coordinates of img in box.
	"""
	h = img.shape[0]
	w = img.shape[1]
	[x,y] = np.meshgrid(np.linspace(box[0],box[1],w),
		                np.linspace(box[2],box[3],h))
	Nx = w*h
	x = np.reshape(x,(Nx))
	y = np.reshape(y,(Nx))
	X = np.vstack([x,y]).T
	return X
	
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
	param['source'] = ['default']
	param['target'] = ['default']
	#### Default target plan base ####
	param['e_eta'] = [1.,0.,0.]
	param['e_xi'] = [0.,0.,1.]
	param['n_plan'] = [0.,-10.,0.]
	
	# If a parameter file is given
	if args.f != '0':
		infile = None
		try:
			infile = open(args.f, "r")
			
			try:
				header = infile.readline().rstrip('\n\r')
		
				if header != 'Reflector parameter file':
					raise ValueError
				
				# Parameters contained in the file
				# will erase the default ones. The
				# others will remain by default.
				for line in infile:
					data = line.rstrip('\n\r').split("\t")
					if data[0] in param:
						param[data[0]] = data[1:]
					else:
						print(data[0],
						' is not a valid parameter and will be ignored.')
			
				e_eta = np.array([float(x) for x in param['e_eta']])
				e_xi = np.array([float(x) for x in param['e_xi']])
				n_plan = np.array([float(x) for x in param['n_plan']])
				
				if param['source'][0] != 'default':
					param['source'] = init_source(param['source'])
				if param['target'][0] != 'default':
					param['target'] = init_target(param['target'])
			
				# Assert the base is orthogonal and 
				# normalize e_eta e_ksi
				assert(np.dot(e_eta,e_xi)==0.)
				cross_prod = np.cross(np.cross(e_eta,e_xi),n_plan)
				assert(np.allclose(cross_prod,np.zeros(3)))
				e_eta /= np.linalg.norm(e_eta)
				e_xi /= np.linalg.norm(e_xi)
				param['e_eta'] = e_eta
				param['e_xi'] = e_xi
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
	
def init_source(source):
	if len(source)==1:
		source = [str(source[0])]
	elif len(source)==3:
		source[0] = str(source[0])
		source[1] = float(source[1])
		source[2] = float(source[2])
	else:
		print ("Error: wrong source parameters")
	return source


def init_target(target):
	if len(target)==1:
		target = [str(target[0])]
	elif len(target)==3:
		target[0] = str(target[0])
		target[1] = float(target[1])
		target[2] = float(target[2])
	elif len(target)==5:
		target[0] = str(target[0])
		for i in range(1,4):
			target[i] = float(target[i])
	else:
		print ("Error: wrong target parameters")
	return target
	
def display_parameters(param):
	for keys in param:
		print(keys,' : ', param[keys])
		
def display_parameters_2(param):
	for key in param:
		if(key=='source' or key=='target'):
			print(str(key), 'name= ', param[key][0])
		else:
			print(keys,' : ', param[keys])
