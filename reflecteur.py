from __future__ import print_function

import sys
import time

sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')
sys.path.append('./lib')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import MongeAmpere as ma
from presolution import *
from preprocessing import *
import geometry as geo

debut = time.clock()

# switch between continue to discrete transformation (XY)
# or discrete to continue (YX)
switch = 'XY'

# Projection plan basis
n_plan = np.array([-10.,0.,0.])
e_eta = np.array([0.,-1,0.])
e_ksi = np.array([0.,0.,1.])

##### Source and target processing #####
X, mu, Y, nu = input_preprocessing(sys.argv, switch)

if switch=='XY':
	gradx, grady = geo.planar_to_gradient(Y[:,0],Y[:,1],e_eta=e_eta,e_ksi=e_ksi,n=n_plan)
	grad = np.vstack([gradx,grady]).T

	t = time.clock() - debut
	print ("Inputs processing:", t, "s")

	##### Optimal Transport problem resolution #####
	psi0 = presolution(grad, X)
	psi = ma.optimal_transport_2(mu, grad, nu, psi0, verbose=True)
	Z,T_Z,psi_Z = eval_legendre_fenchel(mu, grad, psi)
	interpol = make_cubic_interpolator(Z,T_Z,psi_Z,grad=grad)

	t = time.clock() - t
	print ("OT resolution:", t, "s")

	##### Output processing #####
	[gridx, gridy] = np.meshgrid(np.linspace(np.min(Z[:,0]), np.max(Z[:,0]), 100), np.linspace(np.min(Z[:,1]), np.max(Z[:,1]), 100))
	gridx = np.reshape(gridx, (10000))
	gridy = np.reshape(gridy, (10000))

	[gx1, gy1] = interpol.gradient(Z[:,0], Z[:,1])
	[gx, gy] = interpol.gradient(gridx, gridy)


if switch=='YX':
	gradx, grady = geo.planar_to_gradient(X[:,0],X[:,1],e_eta=e_eta,e_ksi=e_ksi,n=n_plan)
	grad = np.vstack([gradx,grady]).T
	
	t = time.clock() - debut
	print ("Inputs processing:", t, "s")
	
	##### Optimal Transport problem resolution #####
	mu_grad = ma.Density_2(grad)
	psi0 = presolution(Y, grad)
	psi = ma.optimal_transport_2(mu_grad, Y, nu, psi0, verbose=True)
	psi_tilde = (Y[:,1] * Y[:,1] + Y[:,0] * Y[:,0]) / 2 - psi
	T = ma.delaunay_2(Y,nu)
	T = tri.Triangulation(Y[:,0],Y[:,1],T)
	interpol = make_cubic_interpolator(Y,T,psi_tilde,grad=grad)

	t = time.clock() - t
	print ("OT resolution:", t, "s")
	
	##### Output processing #####
	[gridx, gridy] = np.meshgrid(np.linspace(np.min(Y[:,0]), np.max(Y[:,0]), 100), np.linspace(np.min(Y[:,1]), np.max(Y[:,1]), 100))
	[gx, gy] = interpol.gradient(Y[:,0], Y[:,1])
	
##### Plot reflector shape #####

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(gridx,gridy,interpol(gridx,gridy), marker=".", lw=0.1)
plt.show()


print ("Execution time:", time.clock() - debut, "s")

s = plt.scatter(X[:,0], X[:,1], c='b', marker=".", lw=0.1)
t = plt.scatter(gx, gy, color='r', marker=".", lw=0.1)
t1 = plt.scatter(gx1, gy1, color='g', marker=".", lw=0.1)
fig = plt.gcf()
ax = plt.gca()
ax.cla() # clear things for fresh plot
fig.gca().add_artist(s)
fig.gca().add_artist(t)
fig.gca().add_artist(t1)
plt.show()
