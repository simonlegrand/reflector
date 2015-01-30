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
import os
import sys
import time

sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')
sys.path.append('./lib')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D

import MongeAmpere as ma
from presolution import *
from preprocessing import *

debut = time.clock()

if len(sys.argv) != 3:
	print ("**** Error : add input files ****");
	exit();

print ("source =", sys.argv[1])
print("target =", sys.argv[2])

N = 5000
##### Target processing #####
t_size = 1.0
mu = input_preprocessing(sys.argv[2], t_size)
X = ma.optimized_sampling_2(mu, N)
Wx = np.ones(N)

##### Source processing #####
densite_source = input_preprocessing(sys.argv[1])
Y = ma.optimized_sampling_2(densite_source,N)
nu = np.ones(N)
nu = (mu.mass() / np.sum(nu)) * nu
tol = 1e-10
assert(np.sum(nu) - mu.mass() < tol), "Different mass in source and in target"

##### Optimal Transport problem resolution #####
psi0 = presolution(Y, nu, X, Wx)
psi = ma.optimal_transport_2(mu, Y, nu, psi0, verbose=True)
psi_tilde = (Y[:,1] * Y[:,1] + Y[:,0] * Y[:,0] - psi) / 2

##### Output processing #####
T = ma.delaunay_2(Y,nu)
T = tri.Triangulation(Y[:,0],Y[:,1],T)
interpol = tri.CubicTriInterpolator(T, psi_tilde, kind='user', dz=(X[:,0],X[:,1]))
[gridx, gridy] = np.meshgrid(np.linspace(np.min(Y[:,0]), np.max(Y[:,0]), 100), np.linspace(np.min(Y[:,1]), np.max(Y[:,1]), 100))
[gx, gy] = interpol.gradient(Y[:,0], Y[:,1])

##### Plot reflector shape #####
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(gridx, gridy, interpol(gridx, gridy), marker=".", lw=0.1)
plt.show()

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
