"""Contains functions used during the preprocessing step"""
import sys
sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')
import os.path
import numpy as np
import MongeAmpere as ma
import matplotlib.pyplot as plt
import scipy as sp

def rgbtogray(imcolor):
	"""Conversion of a rgb picture into a grayscaled picture"""
	imgray = 0.299*imcolor[:,:,0] + 0.587*imcolor[:,:,1] + 0.114*imcolor[:,:,2]
	return imgray
	
def inputPreproc(fileName, xmin=0., ymin=0.):
	extension = os.path.splitext(fileName)[1]
	if extension == ".txt":
		return readData(fileName)

	else:
		return readImage(fileName, xmin, ymin)

def readData(fn):
	"""This function returns x and y points coordinates and
	their associated weights (= 1.0 if not provided),
	from the file fn"""
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
					
			X = np.array([x,y]).T
			dens = ma.Density_2(X)
			return dens
			"""X = ma.Density_2(X).optimized_sampling(N)
			w = np.ones(N, dtype=float)
			return [X, w]"""
			
	except IOError:
		print "Error: can\'t find file or read data"		
	except ValueError:
		print "Error: wrong data type in the file"
	finally:
		f.close()
		
		
def readImage(fn, xmin, ymin):
	"""This function returns points, 
	determined from either a rgb or grayscaled picture"""
	try:
		image = sp.misc.imread(fn)
		dims = np.shape(image)
		if len(dims) == 3:							# If the picture is in RGB
			img = rgbtogray(image)					# Conversion to grayscale

		elif len(dims) !=2:
			raise ValueError
			
		else:
			img = image

		if dims[0] >= dims[1]:						# If the picture is higher than width
			height = 1.0
			width = round(float(dims[1])/float(dims[0]), 2)				
		else:										# If the picture is widther than high
			height = round(float(dims[0])/float(dims[1]), 2)
			width = 1.0

		n = 128 
		nlin = int(n * height)
		ncol = int(n * width)	
		N = nlin * ncol				
		img = sp.misc.imresize(img, (nlin,ncol))		# Image resizing while keeping proportions
		img = np.asarray(img, dtype=float)
		img[img<10.0] = 10.0							# Threshold to avoid empty Laguerre cells on black areas
		img = img/255.0
		
		xmax = xmin + width
		ymax = ymin + height
		box = [xmin, xmax, ymax, ymin]
		dens = ma.Density_2.from_image(img, box)
		return dens
		"""X = dens.optimized_sampling(N)
		w = np.ones(N)			
		return [X, w]"""		 	
		
	except IOError:
		print "Error: can\'t find file or read data"
	except ValueError:
		print "Error: wrong data type in the file"
