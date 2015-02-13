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
		Three layers of color RGB.
		
	Returns
	-------
	imgray : 2D array
		Grayscaled picture
	"""
	imgray = 0.299*imcolor[:,:,0] + 0.587*imcolor[:,:,1] + 0.114*imcolor[:,:,2]
	return imgray
	
def input_preprocessing(arg, switch):
	"""
	Leads to the proper input treatment
	according to arg and switch.
	
	Parameters
	----------
	arg : string
		arg[1] : source
		arg[2] : target
	switch : string
		Indicates the type of transformation
		'XY' = continue->discrete
		'YX' = discrete->continue
		
	Returns
	-------
	dens : Density_2 object
	"""
	if len(arg) != 3:
		source = "square"
		target = "triangle"
		N = 100
		x = np.array([-0.5,-0.5,0.5,0.5])
		y = np.array([-0.5,0.5,-0.5,0.5])
		square = np.array([x,y]).T
			
		x = np.array([0.5,-0.25,-0.25])
		y = np.array([0.,0.433,-0.433])
		tri = np.array([x,y]).T
		
		if switch=='XY':
			mu= ma.Density_2(square)
			square = ma.optimized_sampling_2(mu,N)
			
			dens = ma.Density_2(tri)
			tri = ma.optimized_sampling_2(dens,N)
			nu = np.ones(N) * (mu.mass() / N)
			return square,mu,tri,nu
			
		elif switch=='YX':
			mu= ma.Density_2(tri)
			tri = ma.optimized_sampling_2(mu,N)
			
			dens = ma.Density_2(square)
			square = ma.optimized_sampling_2(dens,N)
			nu = np.ones(N) * (mu.mass() / N)
			return tri,mu,square,nu
		else:
			print("****Error : Wrong switch value")
	else:
		source = arg[1]
		target = arg[2]
	
	extension_source = os.path.splitext(source)[1]
	extension_target = os.path.splitext(target)[1]
	
	Nimg = 5000
	Nshape = 100
	# Modifier et inclure ds une boucle
	# qui parcourt les 2 argv
	if extension_source == ".txt":
		pts_source = read_data(source)
		dens_source = ma.Density_2(pts_source)
		pts_source = ma.optimized_sampling_2(dens_source, Nshape, niter=1)
	else:
		img,box = read_image(source, 1.)
		dens_source = ma.Density_2.from_image(img,box)
		pts_source = ma.optimized_sampling_2(dens_source, Nimg, niter=1)

	if extension_target == ".txt":
		pts_target = read_data(target)
		dens_target = ma.Density_2(pts_target)
		pts_target = ma.optimized_sampling_2(dens_target, Nshape, niter=1)
	else:
		img,box = read_image(target, 1.)
		dens_target = ma.Density_2.from_image(img,box)
		pts_target = ma.optimized_sampling_2(dens_target, Nimg, niter=1)

	if switch=='XY':
		N = len(pts_target)
		w_target = np.ones(N) * (dens_source.mass() / N)
		return pts_source, dens_source, pts_target, w_target

	if switch=='YX':
		N = len(pts_source)
		w_source = np.ones(N) * (dens_target.mass() / N)
		return pts_target, dens_target, pts_source, w_source
		
def read_data(fn):
	"""
	This function returns a Density_2 object,
	determined from points in the file fn.
	
	Parameters
	----------
	fn : string
		File adress
		
	Returns
	-------
	dens : Density_2 object
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
	This function returns a Density_2 object, 
	determined from a picture.
	If the picture is in rgb, it will be grayscaled.
	
	Parameters
	----------
	fn : string
		Picture adress
	size : float
		Length of the largest dimension desired
		
	Returns
	-------
	dens : Density_2 object describing picture density
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

		n = 64
		nlin = int(n * ratio)
		ncol = int(n * ratio)	
		
		img = sp.misc.imresize(img, (nlin,ncol))
		img = np.asarray(img, dtype=float)
		# Threshold to avoid empty Laguerre cells on black areas
		img[img<10.0] = 10.0
		img = img / 255.0
		
		xmin = -(size * ratio) / 2.
		ymin = -(size * ratio) / 2.

		box = [xmin,-xmin,-ymin,ymin]

		return img, box	 	
		
	except IOError:
		print ("Error: can\'t find file or read data")
	except ValueError:
		print ("Error: wrong data type in the file")
		
def eval_legendre_fenchel(mu, Y, psi):
	# on trouve un point dans chaque cellule de Laguerre
	Z = mu.lloyd(Y,psi)[0]
	# par definition, psi^*(z) = min_{y\in Y} |y - z|^2 - psi_y
	# Comme Z[i] est dans la cellule de Laguerre de Y[i], la formule se simplifie:
	psi_star = np.square(Y[:,0] - Z[:,0]) + np.square(Y[:,1] - Z[:,1]) - psi
	T = ma.delaunay_2(Z, psi_star)
	# ensuite, on modifie pour retrouver une fonction convexe \tilde{psi^*} telle
	# que \grad \tilde{\psi^*}(z) = y si z \in Lag_Y^\psi(y)
	psi_star_tilde = np.square(Z[:,0]) + np.square(Z[:,1])/2 - psi_star/2
	return Z,T,psi_star_tilde
	
def make_cubic_interpolator(Z,T,psi,grad):
	T = tri.Triangulation(Z[:,0],Z[:,1],T)
	interpol = tri.CubicTriInterpolator(T, psi, kind='user', dz=(grad[:,0],grad[:,1]))
	return interpol	
