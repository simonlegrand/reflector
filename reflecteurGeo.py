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
from presolution import *
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

debut = time.clock()

if len(sys.argv) != 2:
	print "**** Error : add an image as argument ****";
	exit();

def readData(fileAdress):
	"""
	This function returns a list with x and y points coordinates
	"""
	f = open(fileAdress, 'r')
	try:
		header = f.readline().rstrip('\n\r')
		if header != 'Input file reflector':
			print "Input file error: Wrong file format"
			sys.exit()
		
		x = []
		y = []
		for line in f:
			data = line.rstrip('\n\r').split("\t")
			x.append(float(data[0]))
			y.append(float(data[1]))

		return [x, y]
		
	finally:
		f.close()

##### Target processing #####
shape = np.array(readData(sys.argv[1])).T
Nx = 1000
X = ma.Density_2(shape).optimized_sampling(Nx);
mu = np.ones(Nx)
dens = ma.Density_2(X,mu);

##### Source processing #####
Lsource = 2.0
N = 100
Ndirac = N*N
squareSource = np.array([[0.,0.],[Lsource,0.],[0.,Lsource],[Lsource,Lsource]])
weightsSource = np.array([1., 1., 1., 1.])
Y = ma.Density_2(squareSource).optimized_sampling(Ndirac-4)
Y = np.concatenate((Y, squareSource))
nu = (dens.mass()/Ndirac) * np.ones(Ndirac)

##### Optimal Transport problem resolution #####
psi = ma.optimal_transport_2(dens, Y, nu, presolution(Y, nu, X, mu), verbose=True)
psi_tilde = (Y[:,1]*Y[:,1] + Y[:,0]*Y[:,0] - psi)/2

##### Output processing #####
T = tri.Triangulation(Y[:,0], Y[:,1])
interpol = tri.CubicTriInterpolator(T, psi_tilde)
[gy, gx] = interpol.gradient(Y[:,0], Y[:,1])


"""J = np.logical_or(np.logical_or(np.greater(gy, np.max(X[:,0])), np.greater(gx, np.max(X[:,1]))), np.logical_or(np.less(gy, np.min(X[:,0])), np.less(gx, np.min(X[:,1]))))		#Affichage des points dt le gradient est en dehors de la cible
Z = Y[J]
out = plt.scatter(Z[:,0], Z[:,1]  , color='b', s=0.2)"""

source = plt.scatter(Y[:,0], Y[:,1]  , color='g', s=0.2)
target = plt.scatter(gx, gy, color='r', s=0.2)
fig = plt.gcf()
ax = plt.gca()
ax.cla() # clear things for fresh plot
fig.gca().add_artist(source)
fig.gca().add_artist(target)
plt.show()
