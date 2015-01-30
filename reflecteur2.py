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

# Definition de la position du plan de projection de la cible
theta_0 = np.pi/2
#phi_0 = np.pi/4
phi_0 = 0
d = 5.

# Echantillonnage de la cible dans le repere du plan cible
N = 10000
##### Target processing #####
t_size = 1.0
mu = input_preprocessing(sys.argv[2],t_size)
X = ma.optimized_sampling_2(mu,N)
Wx = np.ones(N)

theta, phi = planar_to_spherical(X[:,0],X[:,1],theta_0,phi_0,d)
gradx, grady = spherical_to_gradient(theta,phi)
g = np.vstack([gradx,grady]).T


##### Source processing #####
densite_source = input_preprocessing(sys.argv[1])
Y = ma.optimized_sampling_2(densite_source,N)
nu = np.ones(N)
nu = (mu.mass() / np.sum(nu)) * nu
tol = 1e-10
assert(np.sum(nu) - mu.mass() < tol), "Different mass in source and in target"


plt.subplot(2,1,1)
t = plt.scatter(grady,gradx, color='r', marker=".", lw=0.1)
plt.subplot(2,1,2)
t2 = plt.scatter(X[:,0],X[:,1], color='r', marker=".", lw=0.1)

plt.show()

psi0 = presolution(Y,nu,g,Wx)
psi = ma.optimal_transport_2(mu,Y,nu,psi0,verbose=True)


