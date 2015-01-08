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

##### Estimation of psi #####
def presolution(img, res, nbdirac, grid):
	n0 = int(res/2.0)
	mu0 = sp.misc.imresize(img, (n0,n0))			
	mu0 = mu0.astype(float)
	dens0 = ma.Density_2.from_image(mu0,[0,1,0,1])
	nu0 = (dens0.mass()/nbdirac) * np.ones(nbdirac)
	return ma.optimal_transport_2(dens0, Y, nu0, verbose=True)


##### Conversion from RGB to grayscale #####
def rgbtogray(imcolor):
	imgray = np.zeros((imcolor.shape[0], imcolor.shape[1]))	# Grayscale conversion
	for rownum in range(len(imcolor)):
   		for colnum in range(len(imcolor[rownum])):
      			imgray[rownum][colnum] = 0.299*imcolor[rownum][colnum][0] + 0.587*imcolor[rownum][colnum][1] + 0.114*imcolor[rownum][colnum][2]
	return imgray



fig = plt.figure()

if len(sys.argv) != 2:
	print "**** Error : add an image as argument ****";
	exit();

##### Picture formating #####
impath = sys.argv[1]
image = sp.misc.imread(impath)
dims = np.shape(image)
print dims

if len(dims) == 3:								# If the picture is in RGB
	img = rgbtogray(image)						# Conversion to grayscale

elif len(dims) !=2:
	print "**** Error : wrong image format ****";
	exit();

else:
	img = image

if dims[0] >= dims[1]:						# If the picture is higher than width
	h = 1.0
	w = round(float(dims[1])/float(dims[0]), 2)					
else:								# If the picture is widther than high
	h = round(float(dims[0])/float(dims[1]), 2)
	w = 1.0

##### Target density calculation #####
n = 200
nlin = int(n * h)
ncol = int(n * w)
mu = sp.misc.imresize(img, (nlin,ncol))			# Image resizing while keeping proportions
mu = mu.astype(float)   								# Transform into a float matrix

"""
plt.imshow(mu, interpolation='nearest', vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
plt.show()
"""

mumoy = np.sum(mu)/(nlin*ncol)    						# Pixels average value
dens = ma.Density_2.from_image(mu,[0,w,0,h]) 			# Density_2 object, contains triangulation points and their density.

##### Source diracs #####
N = 200
Nlin = int(N * h)
Ncol = int(N * w)
Ny = Nlin * Ncol

"""[x,y] = np.meshgrid(np.linspace(0.,1.,N),
                            np.linspace(0.,1.,N)) 		# x and y are coordinates matrix of the grid

x = np.reshape(x,(Ny));
y = np.reshape(y,(Ny));
Y = np.vstack([x,y]).T;"""									# .T=transpose
"""Echantillonage adapte a une source non uniforme"""
rectangle = np.array([[0.,0.],[w,0.],[0.,h],[w,h]])
Y = ma.Density_2(rectangle).optimized_sampling(Ny-4)		# For a non-uniform source
Y = np.concatenate((Y, rectangle))							# Addition of rectangle corners for interpolation

nu = (dens.mass()/Ny) * np.ones(Ny) 						# Diracs weights

##### Optimal Transport problem resolution #####
"""Can be done with or without presolution"""
psi = ma.optimal_transport_2(dens, Y, nu, verbose=True)
#psi = ma.optimal_transport_2(dens, Y, nu, presolution(img, n, Ny, Y), verbose=True);

# Trouver le psi tilde
psi_tilde = (Y[:,0]*Y[:,0] + Y[:,1]*Y[:,1] - psi)/2

# Tracer f= (x, y, psi tilde) et verifier que c'est convexe.
ax = Axes3D(fig)
ax.scatter(Y[:,0], Y[:,1], psi_tilde)
plt.show()

interpol = ipol.CloughTocher2DInterpolator(Y, psi_tilde, tol=1e-6) # psi_tilde interpolation

##### Gradient evaluation #####
nmesh = N * 30
nlinmesh = Nlin * 30
ncolmesh = Ncol * 30
[x,y] = np.meshgrid(np.linspace(0.,w,ncolmesh),
                            np.linspace(0.,h,nlinmesh)) # Retourne deux matrices : Matrices coordonnees de x et de y

Nx = nlinmesh*ncolmesh

x = np.reshape(x,(Nx))
y = np.reshape(y,(Nx))
X = np.vstack([x,y]).T								# Cree une matrice (Nx,2) de coordonnees des noeuds de la grille
I=np.reshape(interpol(X),(nlinmesh,ncolmesh))		# Valeurs de psi_tilde aux noeuds de la grille
"""reshape(a, (nblignes, nbcolonnes)"""
"""interpol(X) est bien convexe"""

[gy, gx] = np.gradient(I, h/nlinmesh, w/ncolmesh)	# On evalue le gradient

gx = np.reshape(gx, (nlinmesh*ncolmesh))			# Remise sous forme de vecteur pour chaque coordonnee du gradient
gy = np.reshape(gy, (nlinmesh*ncolmesh))

threshold = np.ones(Nx)
J = np.logical_and(np.less(gx, threshold*w), np.less(gy, threshold*h))
gx = gx[J]
gy = gy[J]
J = np.logical_and(np.greater(gx, np.zeros(gx.size)), np.greater(gy, np.zeros(gy.size)))
gx = gx[J]
gy = gy[J]
J = np.logical_and(np.isfinite(gx) ,np.isfinite(gy))
gx = gx[J]
gy = gy[J]
print gx, gy
print np.max(gx)
print np.max(gy)

"""OK"""

##### Pseudo ray tracing #####
"""(gx,gy) = "position des rayons reflechis"
# on en deduit des coordonnees de pixels (ix,iy) dans une image de taille nlinpix * ncolpix"""
resolutionfactor = N
nlinpix = int(N * h)						# Number of lines of pixels
ncolpix = int(N * w)						# Number of columns of pixels
ix = np.floor(resolutionfactor*gx).astype(int) 		# Coordonnees x des pixels ou vont tomber les photons de coord gx
iy = np.floor(resolutionfactor*gy).astype(int)		# Coordonnees y	""

data = np.ones(ix.size)
#print np.shape(ix), np.shape(iy), np.shape(data)

M = sparse.coo_matrix((data, (iy,ix)),shape=(nlinpix, ncolpix)).todense()
""",shape=(ncolpix, nlinpix)"""
Mmoy = np.sum(M)/(nlinpix*ncolpix)
M = M/Mmoy*mumoy								# Egalisation de la valeur moyenne des pixels

print "Execution time (seconds):", time.clock() - debut;

plt.imshow(M, interpolation='nearest', vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
#plt.scatter(gx, gy)
plt.show()
