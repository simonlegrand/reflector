from __future__ import print_function

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
import geometry as geo

debut = time.clock()

if len(sys.argv) != 3:
	print ("**** Error : add input files ****");
	exit();

print ("source =", sys.argv[1])
print("target =", sys.argv[2])

##### Target plane position #####
theta_0 = np.pi/2
phi_0 = 0.
d = 10.

##### Source processing #####
X = input_preprocessing(sys.argv[1])
mu = ma.Density_2(X)

t = time.clock() - debut
print ("Source processing:", t, "s")


##### Target processing #####
t_size = 1.0
Y = input_preprocessing(sys.argv[2], t_size)
nu = np.ones(Y.shape[0])
nu = (mu.mass() / np.sum(nu)) * nu
tol = 1e-10
assert(np.sum(nu) - mu.mass() < tol), "Different mass in source and in target"
gradx, grady = geo.planar_to_gradient(Y[:,0],Y[:,1])
grad = np.vstack([gradx,grady]).T

t = time.clock() - t
print ("Target processing:", t, "s")

"""
t = plt.scatter(grady, gradx, color='r', marker=".", lw=0.1)
fig = plt.gcf()
ax = plt.gca()
fig.gca().add_artist(t)
plt.show()"""

def eval_legendre_fenchel(Y, psi):
	# on trouve un point dans chaque cellule de Laguerre
	Z = mu.lloyd(Y,psi)[0]
	# par definition, psi^*(z) = min_{y\in Y} |y - z|^2 - psi_y
	# Comme Z[i] est dans la cellule de Laguerre de Y[i], la formule se simplifie:
	psi_star = np.square(Y[:,0] - Z[:,0]) + np.square(Y[:,1] - Z[:,1]) - psi
	T = ma.delaunay_2(Z, psi_star)
	# ensuite, on modifie pour retrouver une fonction convexe \tilde{psi^*} telle
	# que \grad \tilde{\psi^*}(z) = y si z \in Lag_Y^\psi(y)
	psi_star_tilde = np.square(Z[:,0]) + np.square(Z[:,1])/2 - psi_star/2
	return Z,T,psi_star_tilde
	
def make_cubic_interpolator(Z,T,psi,grad):
	T = tri.Triangulation(Z[:,0],Z[:,1],T)
	interpol = tri.CubicTriInterpolator(T, psi, kind='user', dz=(grad[:,0],grad[:,1]))
	return interpol	


##### Optimal Transport problem resolution #####
psi0 = presolution(grad, X)
psi = ma.optimal_transport_2(mu, grad, nu, psi0, verbose=True)
Z,T_Z,psi_Z = eval_legendre_fenchel(grad, psi)
interpol = make_cubic_interpolator(Z,T_Z,psi_Z,grad=grad)

t = time.clock() - t
print ("OT resolution:", t, "s")


##### Output processing #####
[gridx, gridy] = np.meshgrid(np.linspace(np.min(Z[:,0]), np.max(Z[:,0]), 100), np.linspace(np.min(Z[:,1]), np.max(Z[:,1]), 100))
[gx, gy] = interpol.gradient(Z[:,0], Z[:,1])


##### Plot reflector shape #####

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Z[:,0],Z[:,1],psi_Z, marker=".", lw=0.1)
plt.show()


##### Ray tracing #####

print ("Execution time:", time.clock() - debut, "s")

s = plt.scatter(X[:,0], X[:,1], c='b', marker=".", lw=0.1)
t = plt.scatter(gx, gy, color='r', marker=".", lw=0.1)
fig = plt.gcf()
ax = plt.gca()
ax.cla() # clear things for fresh plot
fig.gca().add_artist(s)
fig.gca().add_artist(t)
plt.show()
