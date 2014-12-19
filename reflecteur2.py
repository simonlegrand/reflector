# PyMongeAmpere
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
sys.path.append('../Pybuild/');
sys.path.append('../Pybuild/lib');
import os
import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.interpolate as ipol
import scipy.optimize as opt
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
import time

debut = time.clock();

fig = plt.figure();


##### Estimation of psi #####
def presolution(img, res, nbdirac, grid):
	n0lin = int(res[0]/2.0)
	n0col = int(res[1]/2.0)
	mu0 = sp.misc.imresize(img, (n0lin,n0col))			
	mu0 = mu0.astype(float)	
	dens0 = ma.Density_2.from_image(mu0,[0,1,0,1]) 	
	
	nu0 = (dens0.mass()/nbdirac) * np.ones(nbdirac)
	return ma.optimal_transport_2(dens0, Y, nu0, verbose=True)


##### Conversion from RGB to grayscale #####
def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2];


if len(sys.argv) != 2:
	print "**** Error : add an image as argument ****";
	exit();

##### Picture formatting #####
impath = sys.argv[1];
image = sp.misc.imread(impath);	
dims = np.shape(image);

if len(dims) == 3:						# If the picture is in RGB
	img = np.zeros((image.shape[0], image.shape[1]))	# Grayscale conversion
	for rownum in range(len(image)):
   		for colnum in range(len(image[rownum])):
      			img[rownum][colnum] = weightedAverage(image[rownum][colnum]);

elif len(dims) !=2:
	print "**** Error : wrong image format ****";
	exit();

else:
	img = image

if dims[0] >= dims[1]:						# If the picture is higher than width
	fraclin = 1
	fraccol = round(float(dims[1])/float(dims[0]), 2)					
else:								# If the picture is widther than high
	fraclin = round(float(dims[0])/float(dims[1]), 2)
	fraccol = 1


##### Target density calculation #####
n = 128								# Higher dimension number of pixels
nlin = int(n*fraclin)
ncol = int(n*fraccol)
mu = sp.misc.imresize(img, (nlin,ncol));			# Image resizing while keeping proportions
mu = mu.astype(float);   					# Transform into a float matrix
mumoy = np.sum(mu)/(nlin*ncol);	    				# Pixels average value
dens = ma.Density_2.from_image(mu,[0,1,0,1]) 			# Density_2 object, contains triangulation points and their density.


##### Source diracs #####
"""Echantillonage regulier pour une source carre uniforme"""
N = 128

[x,y] = np.meshgrid(np.linspace(0.,1.,N),
                            np.linspace(0.,1.,N)) 		# x and y are coordinates matrix of the grid

Ny = N*N;

x = np.reshape(x,(Ny));
y = np.reshape(y,(Ny));
Y = np.vstack([x,y]).T; # .T=transpose"""

"""Echantillonage adapte a une source non uniforme"""
"""square = np.array([[0.,0.],[0.,1.],[1.,1.],[1.,0.]]);
Y = ma.Density_2(square).optimized_sampling(Ny);""" 		# Non uniform source

nu = (dens.mass()/Ny) * np.ones(Ny); 				# Diracs weights, sum equal to the mass of the picture

##### Optimal Transport problem resolution #####
"""Can be done with or without presolution"""
#psi = ma.optimal_transport_2(dens, Y, nu, verbose=True);
psi = ma.optimal_transport_2(dens, Y, nu, presolution(img, [nlin,ncol], Ny, Y), verbose=True);

"""[C,m] = ma.lloyd_2(dens,Y,psi); 			# C = Laguerre diagram(Y, psi) centroids; m = cells mass
plt.scatter(C[:,0],C[:,1],s=1);
plt.show()"""


# Trouver le psi tilde (verifier dans Cgal l'expression des cellules de laguerre)
psi_tilde = (Y[:,0]*Y[:,0] + Y[:,1]*Y[:,1] - psi)/2;

# Tracer f= (x, y, psi tilde) et verifier que c'est convexe.
ax = Axes3D(fig)
ax.scatter(Y[:,0], Y[:,1], psi_tilde);
plt.show()

interpol = ipol.CloughTocher2DInterpolator(Y, psi_tilde, tol=1e-6) # psi_tilde interpolation
#print interpol(Y);


##### Evaluation du gradient d'interpol sur une grille fine #####
nmesh = 30*N;
[x,y] = np.meshgrid(np.linspace(0.,1.,nmesh),
                            np.linspace(0.,1.,nmesh));
Nx = nmesh*nmesh;

x = np.reshape(x,(Nx));
y = np.reshape(y,(Nx));
X = np.vstack([x,y]).T;					# Cree une matrice (Nx,2) de coordonnees des noeuds de la grille
I=np.reshape(interpol(X),(nmesh,nmesh));		# Valeurs de psi_tilde aux noeuds de la grille

[gx, gy] = np.gradient(I, 1./nmesh, 1./nmesh);		# On evalue le gradient

gx = np.reshape(gx, (Nx));				# Remise sous forme de vecteur pour chaque coordonnee du gradient
gy = np.reshape(gy, (Nx));

print np.min(gx)
print np.min(gy)

##### Pseudo ray tracing #####
"""(gx,gy) = "position des rayons reflechis"
# on en deduit des coordonnees de pixels (ix,iy) dans une image de taille npix * npix"""
npix = N
#pixl = int(npix*fraclin)
#npixc = int(npix*fraccol)
ix = np.absolute(np.floor(npix*gx)); 		# Coordonnees x des pixels ou vont tomber les photons de coord gx
iy = np.absolute(np.floor(npix*gy));		# Coordonnees y	""

print np.min(iy)
print np.min(ix)

data = np.ones(nmesh*nmesh);
print np.shape(ix), np.shape(iy), np.shape(data)

M = sparse.coo_matrix((data, (ix,iy)), shape=(npix,npix)).todense()

Mmoy = np.sum(M)/(npix*npix);
M = M/Mmoy*mumoy;					# Egalisation de la valeur moyenne des pixels

print "Execution time (seconds):", time.clock() - debut;

plt.imshow(M, interpolation='nearest', vmin=0, vmax=255, cmap=plt.get_cmap('gray'));
#plt.scatter(gx, gy);
print pouet
plt.show()
