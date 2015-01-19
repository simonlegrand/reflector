#TODO: Probleme d'egalisation des masses
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
sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')
sys.path.append('./Functions')
import os
from presolution import *
from preprocessing import *
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

if len(sys.argv) != 3:
	print "**** Error : add source files ****";
	exit();

##### Target processing #####
[X, mu] = inputPreproc(sys.argv[2])
print mu
dens = ma.Density_2(X,mu)

##### Source processing #####
[Y, nu] = inputPreproc(sys.argv[1])
#nu = (dens.mass()/np.shape(Y)[0]) * np.ones(np.shape(Y)[0])
#nuMoy = np.sum(nu) / np.shape(Y)[0]

print dens.mass(), nuMoy

##### Optimal Transport problem resolution #####
psi = ma.optimal_transport_2(dens, Y, nu, presolution(Y, nu, X, mu), verbose=True)
psi_tilde = (Y[:,1]*Y[:,1] + Y[:,0]*Y[:,0] - psi)/2

##### Output processing #####
T = tri.Triangulation(Y[:,0], Y[:,1])
interpol = tri.CubicTriInterpolator(T, psi_tilde)
[gx, gy] = interpol.gradient(Y[:,0], Y[:,1])

"""J = np.logical_or(np.logical_or(np.greater(gy, np.max(X[:,0])), np.greater(gx, np.max(X[:,1]))), np.logical_or(np.less(gy, np.min(X[:,0])), np.less(gx, np.min(X[:,1]))))		#Affichage des points dont le gradient est en dehors de la cible pour une source CARRE!
Z = Y[J]
out = plt.scatter(Z[:,0], Z[:,1]  , color='b', s=0.2)"""

s = plt.scatter(Y[:,0], Y[:,1]  , color='g', s=0.2)
t = plt.scatter(gx, gy, color='r', s=0.2)
fig = plt.gcf()
ax = plt.gca()
ax.cla() # clear things for fresh plot
fig.gca().add_artist(s)
fig.gca().add_artist(t)
plt.show()
