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

import MongeAmpere as ma
import misc

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
	source = parameters['source']
	target = parameters['target']
	
	#### Source processing ####
	if source == 'default':
		
		x = np.array([-0.5,-0.5,0.5,0.5])
		y = np.array([-0.5,0.5,-0.5,0.5])
		X = np.array([x,y]).T
		mu = ma.Density_2(X)
		
	else:
		extension_source = os.path.splitext(source)[1]
		
		if extension_source == ".txt":
			X, X_w = read_data(source)
			mu = ma.Density_2(X, X_w)
		
		else:
			img,box = read_image(source, parameters['s_size'])
			# Null pixels are removed
			X = regular_sampling(img,box)
			img = np.reshape(img, np.shape(X)[0])
			zero = np.zeros(np.shape(img))
			J = np.greater(img,zero)
			img = img[J]
			X = X[J]
			mu = ma.Density_2(X,img)
			#mu = ma.Density_2.from_image(img,box)
		
	#### Target processing ####
	if target == 'default':
		# Number of diracs for a shape
		Ndiracs = 10000
		#x = np.array([0.5,-0.25,-0.25])
		#y = np.array([0.,0.433,-0.433])
		x = np.array([0.,-1.,-0.75, 0.75, 1.])
		y = np.array([1.5,0.25,-0.5, -0.5, 0.25])
		tri = np.array([x,y]).T
		dens_target = ma.Density_2(tri)
		Y = ma.optimized_sampling_2(dens_target,Ndiracs)
		nu = np.ones(Ndiracs) * (mu.mass() / Ndiracs)
		
	else:
		extension_target = os.path.splitext(target)[1]

		if extension_target == ".txt":
			Y, nu = read_data(target)
			nu = nu * (mu.mass() / sum(nu))
		
		else:
			# For a picture, number of dirac equals
			# number of pixels (set in read_image())
			img,box = read_image(target, parameters['t_size'])
			Y = regular_sampling(img,box)
			N = len(Y)
			nu = np.reshape(img,(N))
			
			# Null pixels are removed
			zero = np.zeros(N)
			J = np.greater(nu,zero)
			Y = Y[J]
			nu = nu[J]
			
			# Mass equalization
			nu = nu * (mu.mass()/sum(nu))

	return mu, Y, nu


def read_data(fn):
	"""
	This function returns points coordinates
	contained in file fn.
	
	Parameters
	----------
	fn : string
		File adress
		
	Returns
	-------
	X : 2D array
		Points coordinates of the polygon
	z : 1D array
		Weights associated to points X
	"""
	try:
		infile = open(fn, "r")
		
		try:
			header = infile.readline().rstrip('\n\r')
		
			if header != 'Reflector input file':
				raise ValueError
		
			else:
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
			
				return X, z
				
		finally:
			infile.close()
			
	except IOError:
		print ("Error: can\'t find", fn)
		sys.exit()
		
	except ValueError:
		print ("Error: wrong data type in", fn)
		sys.exit()

def read_image(fn, size):
	"""
	This function returns pixels value between 0 and 1.
	If the picture is in rgb, it is grayscaled.
	
	Parameters
	----------
	fn : string
		Picture adress
	size : float
		Length of the largest dimension desired
		
	Returns
	-------
	img : 2D array
		Pixels values
	box : 1D array
		Bounding box of the picture
	"""
	try:
		image = sp.misc.imread(fn)

		dims = np.shape(image)
		if len(dims) == 3:
			img = rgb_to_gray(image)

		elif len(dims) != 2:
			raise ValueError

		else:
			img = image

		ratio = float(dims[0]) / dims[1]

		nlin = 100
		ncol = int(nlin / ratio)	

		img = sp.misc.imresize(img, (nlin,ncol))
		#img = sp.misc.imrotate(img,-90)	# Optionnal
		img = np.asarray(img, dtype=float)
		img = img / 255.0
		
		# Center the image
		xmin = -(size / ratio) / 2.
		ymin = -size / 2.
		box = [xmin,-xmin,-ymin,ymin]
		
		return img, box
	
	except IOError:
		print ("Error: can\'t find", fn)
		sys.exit()
	
def regular_sampling(img,box):
	"""
	This function returns regular sampling of img,
	corresponding to pixel coordinates.
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
	param['source'] = 'default'
	param['target'] = 'default'
	#### Default target plan base ####
	param['e_eta'] = [1.,0.,0.]
	param['e_ksi'] = [0.,0.,1.]
	param['n_plan'] = [0.,-10.,0.]
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
					if param['source'] != 'default':
						param['source'] = str(param['source'][0])
					if param['target'] != 'default':
						param['target'] = str(param['target'][0])
				
					# Assert the base is orthogonal and 
					# normalize e_eta e_ksi
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
