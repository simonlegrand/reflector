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
sys.path.append('./Functions')
import os
import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.interpolate as ipol
import scipy.optimize as opt
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geometry2D import *
from presolution import *
import time

debut = time.clock()

def argcheck(argv):
	"""
	This function checks if two shapes have been given in argument.
	"""
	if len(argv) != 3:
		print "**** Error : Source or target shape missing ****";
		sys.exit();

argcheck(sys.argv)
source = sys.argv[1]
target = sys.argv[2]

##### Target density #####
if target == "square":
	ltarget = 1.0
	xmin = 3.0
	ymin = 2.0
	squareTarget = np.array([[xmin,ymin],[xmin+ltarget,ymin],[xmin,ymin+ltarget],[xmin+ltarget,ymin+ltarget]])
	weightsTarget = np.array([1., 1., 1., 1.])
	mumoy = np.sum(weightsTarget)/len(weightsTarget)
	dens = ma.Density_2(squareTarget, weightsTarget)

if target == "rectangleVertical":
	width = 1.0
	height = 2.0
	rectangle = np.array([[0.,-0.5],[width,-0.5],[0.,-0.5+height],[width,-0.5+height]])
	mumoy = 1.0
	dens = ma.Density_2(rectangle)

if target == "rectangleHorizontal":
	width = 2.0
	height = 1.0
	rectangle = np.array([[-0.5,0.],[-0.5+width,0.],[-0.5,height],[-0.5+width,height]])
	mumoy = 1.0
	dens = ma.Density_2(rectangle)
	
	
if target =="disk":
	Nx = 30;
	t = np.linspace(0,2*np.pi,Nx+1);
	t = t[0:Nx]
	disk = 2*np.vstack([np.cos(t),np.sin(t)]).T;
	mumoy = 1.0
	dens = ma.Density_2(disk)
	
##### Source diracs #####	
if source == "square":
	print "la source est un carre"
	Lsource = 1.0
	N = 100
	Ndirac = N*N
	squareSource = np.array([[0.,0.],[Lsource,0.],[0.,Lsource],[Lsource,Lsource]])
	weightsSource = np.array([1., 1., 1., 1.])
	Y = ma.Density_2(squareSource).optimized_sampling(Ndirac-4)
	Y = np.concatenate((Y, squareSource))
	nu = (dens.mass()/Ndirac) * np.ones(Ndirac)
	"""
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(Y[:,0], Y[:,1], 1.0)
	plt.show()
	"""


##### Presolution #####
"""
Similitude qui envoie la cible dans la source
et calcule un premiere iteration psi
A tester: envoyer le barycentre de la source sur celui de la cible puis 
retrecir la source jusqu'a ce que son enveloppe convexe soit incluse
dans celle de la cible.
"""
	
##### Optimal Transport problem resolution #####
"""Can be done with or without presolution"""
psi = ma.optimal_transport_2(dens, Y, nu, presolution(Y, nu, squareTarget, weightsTarget), verbose=True)
psi_tilde = (Y[:,1]*Y[:,1] + Y[:,0]*Y[:,0] - psi)/2

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Y[:,0], Y[:,1], psi_tilde)
plt.show()

interpol = ipol.CloughTocher2DInterpolator(Y, psi_tilde, tol=1e-6)


##### Gradient calculation #####
nmesh = 30*N
[x,y] = np.meshgrid(np.linspace(0.,Lsource,nmesh),
                            np.linspace(0.,Lsource,nmesh))

Nx = nmesh*nmesh

x = np.reshape(x,(Nx))
y = np.reshape(y,(Nx))
X = np.vstack([x,y]).T								# Cree une matrice (Nx,2) de coordonnees des noeuds de la grille

I=np.reshape(interpol(X),(nmesh,nmesh))
[gy, gx] = np.gradient(I, Lsource/nmesh, Lsource/nmesh)
print interpol(X)
print np.max(gy)

gx = np.reshape(gx, (nmesh*nmesh))			# Remise sous forme de vecteur pour chaque coordonnee du gradient
gy = np.reshape(gy, (nmesh*nmesh))

threshold = np.ones(Nx)
J = np.logical_and(np.less(gx, threshold), np.less(gy, threshold))
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

##### Pseudo ray tracing #####
if target == "square":
	npix = N						# Number of columns of pixels
	ix = np.floor(npix*gx).astype(int) 		# Coordonnees x des pixels ou vont tomber les photons de coord gx
	iy = np.floor(npix*gy).astype(int)		# Coordonnees y	""

	data = np.ones(ix.size)
	#print np.shape(ix), np.shape(iy), np.shape(data)

	M = sparse.coo_matrix((data, (iy,ix)),shape=(npix, npix)).todense()

	Mmoy = np.sum(M)/(npix*npix)
	M = M/Mmoy*mumoy								# Egalisation de la valeur moyenne des pixels

if target == "rectangleVertical":
	nlinpix = int(N * height)						# Number of lines of pixels
	ncolpix = int(N * width)						# Number of columns of pixels
	ix = np.floor(ncolpix*gx).astype(int) 		# Coordonnees x des pixels ou vont tomber les photons de coord gx
	iy = np.floor(nlinpix*gy).astype(int)		# Coordonnees y	""

	data = np.ones(ix.size)
	#print np.shape(ix), np.shape(iy), np.shape(data)

	M = sparse.coo_matrix((data, (iy,ix)),shape=(nlinpix, ncolpix)).todense()
	""",shape=(ncolpix, nlinpix)"""
	Mmoy = np.sum(M)/(nlinpix*ncolpix)
	M = M/Mmoy*mumoy
	
if target == "rectangleHorizontal":
	nlinpix = int(N * height)						# Number of lines of pixels
	ncolpix = int(N * width)						# Number of columns of pixels
	ix = np.floor(ncolpix*gx).astype(int) 		# Coordonnees x des pixels ou vont tomber les photons de coord gx
	iy = np.floor(nlinpix*gy).astype(int)		# Coordonnees y	""

	data = np.ones(ix.size)
	#print np.shape(ix), np.shape(iy), np.shape(data)

	M = sparse.coo_matrix((data, (iy,ix)),shape=(nlinpix, ncolpix)).todense()
	""",shape=(ncolpix, nlinpix)"""
	Mmoy = np.sum(M)/(nlinpix*ncolpix)
	M = M/Mmoy*mumoy

print "Execution time (seconds):", time.clock() - debut;

plt.imshow(M, interpolation='nearest', vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
#plt.scatter(gx, gy)
plt.show()

sys.exit()

"""
fig = plt.figure()
plt.imshow(mu, interpolation='nearest', vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
plt.show()
"""

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Y[:,0], Y[:,1], 1.0)
plt.show()
