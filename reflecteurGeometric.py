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
from pyhull.convex_hull import ConvexHull
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
	square = np.array([[0.,0.],[ltarget,0.],[0.,ltarget],[ltarget,ltarget]])
	dens = ma.Density_2(square)

if target == "triangle":
	ltarget = 6.0
	triangle = np.array([[0.,0.],[ltarget,0.],[ltarget/2.0,ltarget*np.sin(sp.pi/3.0)]])
	dens = ma.Density_2(triangle)
	
##### Source diracs #####
if source == "triangle":
	print "la source est un triangle"
	Lsource = 1.0
	Ndirac = 1000
	triangle = np.array([[0.,0.],[Lsource,0.],[Lsource/2.0,Lsource*np.sin(sp.pi/3.0)]])
	Y = ma.Density_2(triangle).optimized_sampling(Ndirac-3)
	Y = np.concatenate((Y, triangle))
	nu = (dens.mass()/Ndirac) * np.ones(Ndirac)
	
if source == "square":
	print "la source est un carre"
	Lsource = 1.0
	Ndirac = 1000
	square = np.array([[1.,0.],[Lsource+1.,0.],[1.,Lsource],[Lsource+1.,Lsource]])
	Y = ma.Density_2(square).optimized_sampling(Ndirac-4)
	Y = np.concatenate((Y, square))
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
A tester, envoyer le barycentre de la source sur celui de la cible puis 
retrecir la source jusqu'a ce que son enveloppe convexe soit incluse
dans celle de la cible.
"""
##### Optimal Transport problem resolution #####
"""Can be done with or without presolution"""
psi = ma.optimal_transport_2(dens, Y, nu, verbose=True)
psi_tilde = (Y[:,1]*Y[:,1] + Y[:,0]*Y[:,0] - psi)/2
"""
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Y[:,0], Y[:,1], psi_tilde)
plt.show()
"""
interpol = ipol.CloughTocher2DInterpolator(Y, psi_tilde, tol=1e-6)


##### Gradient calculation #####
nmesh = 10
[x,y] = np.meshgrid(np.linspace(1.,Lsource+1.,nmesh),
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
J = np.logical_and(np.greater(gx, np.zeros(gx.size)), np.greater(gy, np.zeros(gy.size)))
gx = gx[J]
gy = gy[J]
J = np.logical_and(np.isfinite(gx) ,np.isfinite(gy))
gx = gx[J]
gy = gy[J]
print gx, gy
print np.max(gx)
print np.max(gy)

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
