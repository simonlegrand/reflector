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

from __future__ import print_function
import sys
sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')
sys.path.append('./lib')
import os
from presolution import *
from preprocessing import *
import MongeAmpere as ma
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
import time

debut = time.clock()

if len(sys.argv) != 3:
	print ("**** Error : add source files ****");
	exit();

print ("source =", sys.argv[1])
print("target =", sys.argv[2])

N = 10000
##### Target processing #####
mu = inputPreproc(sys.argv[2], 1.0, 1.0)
X = ma.optimized_sampling_2(mu, N)
Wx = np.ones(N)

##### Source processing #####
densSource = inputPreproc(sys.argv[1])
Y = ma.optimized_sampling_2(densSource, N)
nu = np.ones(N)
nu = (mu.mass()/np.sum(nu)) * nu
tol = 1e-10
assert(np.sum(nu) - mu.mass() < tol), "Different mass in source and in target"

##### Optimal Transport problem resolution #####
psi0 = presolution(Y, nu, X, Wx)
psi = ma.optimal_transport_2(mu, Y, nu, psi0, verbose=True)
psi_tilde = (Y[:,1]*Y[:,1] + Y[:,0]*Y[:,0] - psi)/2

##### Output processing #####
T = ma.delaunay_2(Y, nu)
T = tri.Triangulation(Y[:,0], Y[:,1], T)
interpol = tri.CubicTriInterpolator(T, psi_tilde, kind='user', dz=(X[:,0], X[:,1]))
[gridx, gridy] = np.meshgrid(np.linspace(np.min(Y[:,0]), np.max(Y[:,0]), 100), np.linspace(np.min(Y[:,1]), np.max(Y[:,1]), 100))
[gx, gy] = interpol.gradient(Y[:,0], Y[:,1])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(gridx, gridy, interpol(gridx, gridy), marker=".", lw=0.1)
plt.show()

"""J = np.logical_or(np.logical_or(np.greater(gy, np.max(X[:,0])), np.greater(gx, np.max(X[:,1]))), np.logical_or(np.less(gy, np.min(X[:,0])), np.less(gx, np.min(X[:,1]))))		#Affichage des points dont le gradient est en dehors de la cible pour une source CARRE!
Z = Y[J]
out = plt.scatter(Z[:,0], Z[:,1]  , color='b', s=0.2)"""

print ("Execution time:", time.clock() - debut)

z = np.max(nu) - nu/np.max(nu)
s = plt.scatter(Y[:,0], Y[:,1], c='b', marker=".", lw=0.1)
t = plt.scatter(gx, gy, color='r', marker=".", lw=0.1)
fig = plt.gcf()
ax = plt.gca()
ax.cla() # clear things for fresh plot
fig.gca().add_artist(s)
fig.gca().add_artist(t)
plt.show()
