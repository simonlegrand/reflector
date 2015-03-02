"""Contains functions used during the input processing step."""
from __future__ import print_function
import sys
sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')
import os.path
import numpy as np
import MongeAmpere as ma
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy as sp

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
	
def input_preprocessing(arg):
	"""
	Leads to the proper input treatment
	according to arg and switch.
	
	Parameters
	----------
	arg : string
		arg[1] : source file name
		arg[2] : target file name
		
	Returns
	-------
	X : 2D array
	mu : Density_2 object
	Y : 2D array
	nu : 1D array
	"""
	Nimg = 100000	# Number of diracs for a picture
	Nshape = 20000	# Number of diracs for a shape
	
	if len(arg) != 3:	# If no argument is passed
		
		x = np.array([-0.5,-0.5,0.5,0.5])
		y = np.array([-0.5,0.5,-0.5,0.5])
		square = np.array([x,y]).T
			
		x = np.array([0.5,-0.25,-0.25])
		y = np.array([0.,0.433,-0.433])
		tri = np.array([x,y]).T
		
		mu = ma.Density_2(square)
		X = ma.optimized_sampling_2(mu,Nshape)
		
		dens_target = ma.Density_2(tri)
		Y = ma.optimized_sampling_2(dens_target,Nshape)
		nu = np.ones(Nshape) * (mu.mass() / Nshape)
		return X,mu,Y,nu
			
	source = arg[1]
	target = arg[2]
	extension_source = os.path.splitext(source)[1]
	extension_target = os.path.splitext(target)[1]

	if extension_source == ".txt":
		X = read_data(source)
		mu = ma.Density_2(X)
		X = ma.optimized_sampling_2(mu, Nshape, niter=1)
	else:
		img,box = read_image(source, 1.)
		mu = ma.Density_2.from_image(img,box)
		if Nimg > 10000:
			X = regular_sampling(img,box)
		else:
			X = ma.optimized_sampling_2(mu, Nimg, niter=1)

	if extension_target == ".txt":
		Y = read_data(target)
		dens_target = ma.Density_2(Y)
		Y = ma.optimized_sampling_2(dens_target, Nshape, niter=2)
		nu = np.ones(Nshape) * (mu.mass() / Nshape)
	else:
		img,box = read_image(target, 1.)
		dens_target = ma.Density_2.from_image(img,box)
		if Nimg > 10000:
			Y = regular_sampling(img,box)
			N = len(Y)
			nu = np.reshape(img,(N))
			zero = np.zeros(nu.shape)
			J = np.greater(nu,zero)
			Y = Y[J]
			nu = nu[J]
			nu = nu * (mu.mass()/sum(nu))
		else:
			Y = ma.optimized_sampling_2(dens_target, Nimg, niter=2)
			nu = np.ones(Nimg) * (mu.mass() / Nimg)

	return X, mu, Y, nu


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
		header = f.readline().rstrip('\n\r')
		
		if header != 'Input file reflector':
			raise ValueError
		
		else:
			x = []
			y = []
			for line in f:
				data = line.rstrip('\n\r').split("\t")
				x.append(float(data[0]))
				y.append(float(data[1]))
				
			x = np.asarray(x)
			y = np.asarray(y)
			
			# Recentering of points around origin		
			xmax = np.max(x)
			xmin = np.min(x)
			ymax = np.max(y)
			ymin = np.min(y)
			x = x - xmin - (xmax - xmin) / 2
			y = y - ymin - (ymax - ymin) / 2
			X = np.array([x,y]).T	

			return X
			
	except IOError:
		print ("Error: can\'t find file or read data")
	except ValueError:
		print ("Error: wrong data type in the file")
	finally:
		f.close()

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
		
		ratio = dims[0] / dims[1]

		nlin = 512
		ncol = int(nlin / ratio)	
		
		img = sp.misc.imresize(img, (nlin,ncol))
		img = np.asarray(img, dtype=float)
		img = img / 255.0
		xmin = -(size / ratio) / 2.
		ymin = -size / 2.

		box = [xmin,-xmin,-ymin,ymin]
		#plt.imshow(img)
		#plt.show()
		return img, box	 	
		
	except IOError:
		print ("Error: can\'t find file or read data")
	except ValueError:
		print ("Error: wrong data type in the file")
		
		
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
