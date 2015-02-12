"""Contains functions used during the input processing step."""
from __future__ import print_function
import sys
sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')
import os.path
import numpy as np
import MongeAmpere as ma
import matplotlib.pyplot as plt
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
	
def input_preprocessing(fileName, size=0.):
	"""
	Leads to the proper treatment according to the
	nature of the file.
	
	Parameters
	----------
	fn : string
		File adress
	size : float
		Only for a picture
		
	Returns
	-------
	dens : Density_2 object
	"""
	extension = os.path.splitext(fileName)[1]
	if extension == ".txt":
		return read_data_2(fileName)

	else:
		return read_image_2(fileName,size)

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

			# Eventually add weights in argument of Density_2
			dens = ma.Density_2(X)
			return dens
			
	except IOError:
		print ("Error: can\'t find file or read data")
	except ValueError:
		print ("Error: wrong data type in the file")
	finally:
		f.close()
	
def read_data_2(fn):
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
			dens = ma.Density_2(X)
			X = ma.optimized_sampling_2(dens,100)

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

		n = 128
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
		dens = ma.Density_2.from_image(img,box)
		return dens		 	
		
	except IOError:
		print ("Error: can\'t find file or read data")
	except ValueError:
		print ("Error: wrong data type in the file")
		
def read_image_2(fn, size):
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
		dens = ma.Density_2.from_image(img,box)
		X = ma.optimized_sampling_2(dens, 5000,niter=1)
		return X	 	
		
	except IOError:
		print ("Error: can\'t find file or read data")
	except ValueError:
		print ("Error: wrong data type in the file")
