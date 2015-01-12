# Reflecteur
# Copyright (C) 2014 Quentin Merigot, CNRS
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import sys
sys.path.append('../Pybuild/')
sys.path.append('../Pybuild/lib')
import os
import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.interpolate as ipol
import scipy.optimize as opt
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

debut = time.clock()

def rgbtogray(imcolor):
	"""
	This function converts a RGB picture into a grayscaled picture
	"""
	imgray = 0.299*imcolor[:,:,0] + 0.587*imcolor[:,:,1] + 0.114*imcolor[:,:,2]
	return imgray


def picturecheck(argv):
	"""
	This function checks the arguments given by the user
	and convert an RGB picture to grayscale
	"""
	if len(argv) != 2:
		print "**** Error : add an image as argument ****";
		exit();

	impath = argv[1]
	image = sp.misc.imread(impath)
	dims = np.shape(image)

	if len(dims) == 3:								# If the picture is in RGB
		img = rgbtogray(image)						# Conversion to grayscale

	elif len(dims) !=2:
		print "**** Error : wrong image format ****";
		exit();

	else:
		img = image

	return img


mu = picturecheck(sys.argv)
dims = np.shape(mu)
if dims[0] >= dims[1]:							# If the picture is higher than width
	height = 1.0
	width = round(float(dims[1])/float(dims[0]), 2)					
else:											# If the picture is widther than high
	height = round(float(dims[0])/float(dims[1]), 2)
	width = 1.0

n = 128 
nlin = int(n * height)
ncol = int(n * width)
mu = sp.misc.imresize(mu, (nlin,ncol))			# Image resizing while keeping proportions
print "Target image size: ", np.shape(mu)
mu = mu.astype(float)   						# Transform into a float matrix
mumoy = np.sum(mu)/(nlin*ncol)    				# Pixels average value


##### Target density calculation #####
"""
fig = plt.figure()
plt.imshow(mu, interpolation='nearest', vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
plt.show()
"""
dens = ma.Density_2.from_image(mu,[0,width,0,height]) 			# Density_2 object, contains triangulation points and their density.

##### Source diracs #####
N = 100
Nlin = int(N * height)
Ncol = int(N * width)
Ny = Nlin * Ncol


width = 1.0
Ny = 100
triangle = np.array([[0.,0.],[width,0.],[width/2.0,width*np.sin(sp.pi/3.0)]])
Y = ma.Density_2(triangle).optimized_sampling(Ny-3)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Y[:,0], Y[:,1], 1.0)
plt.show()


