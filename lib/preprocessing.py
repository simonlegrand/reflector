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
	Leads to the proper input treatment
	according to arg.
	
	Parameters
	----------
	parameters : dictionnary
		Contains all the parameters
		
	Returns
	-------
	X : 2D array
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
		x = np.array([0.5,-0.25,-0.25])
		y = np.array([0.,0.433,-0.433])
		tri = np.array([x,y]).T
		dens_target = ma.Density_2(tri)
		Y = ma.optimized_sampling_2(dens_target,Ndiracs)
		nu = np.ones(Ndiracs) * (mu.mass() / Ndiracs)
		
	else:
		extension_target = os.path.splitext(target)[1]

		if extension_target == ".txt":
			Y, nu = read_data(target)
			dens_target = ma.Density_2(Y, nu)
			nu = nu * (mu.mass() / sum(nu))
		
		else:
			# For a picture, number of dirac equals
			# number of pixels (set in read_image())
			img,box = read_image(target, parameters['t_size'])
			dens_target = ma.Density_2.from_image(img,box)
			Y = regular_sampling(img,box)
			N = len(Y)
			nu = np.reshape(img,(N))
			
			# Null pixels are removed
			zero = np.zeros(N)
			J = np.greater(nu,zero)
			Y = Y[J]
			nu = nu[J]
			
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
	"""
	try:
		f = open(fn, "r")
		
		try:
			header = f.readline().rstrip('\n\r')
		
			if header != 'Reflector input file':
				raise ValueError
		
			else:
				x = []
				y = []
				z = []
				for line in f:
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
				
				# Recentering of points around origin		
				xmax = np.max(x)
				xmin = np.min(x)
				ymax = np.max(y)
				ymin = np.min(y)
				x = x - xmin - (xmax - xmin) / 2
				y = y - ymin - (ymax - ymin) / 2
				X = np.array([x,y]).T
			
				return X, z
				
		finally:
			f.close()
			
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

		nlin = 316
		ncol = int(nlin / ratio)	

		img = sp.misc.imresize(img, (nlin,ncol))
		img = np.asarray(img, dtype=float)
		img = img / 255.0
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
