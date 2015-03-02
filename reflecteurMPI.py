#!/usr/bin/python
from __future__ import print_function

import sys
import time

sys.path.append('./lib')
sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpi4py.MPI as mpi

import MongeAmpere as ma
from presolution import *
from preprocessing import *
import geometry as geo
import rayTracingMPI as ray

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Target plan basis
n_plan = np.array([-5.,0.,0.])
e_eta = np.array([0.,-1,0.])
e_ksi = np.array([0.,0.,1.])

# Incident rays direction
s1 = np.array([0.,0.,1.])

debut = time.clock()

##### Source and target processing #####
X, mu, Y, nu = input_preprocessing(sys.argv)
print('Number of diracs: ', len(nu))
gradx, grady = geo.planar_to_gradient(Y[:,0], Y[:,1], e_eta=e_eta, e_ksi=e_ksi, n=n_plan)
grad = np.vstack([gradx,grady]).T
psi = np.empty(len(gradx))

if rank == 0:
	t = time.clock() - debut
	print ("Inputs processing:", t, "s")

	##### Optimal Transport problem resolution #####
	psi0 = presolution(grad, X)
	psi = ma.optimal_transport_2(mu, grad, nu, psi0, verbose=True)

	t = time.clock() - t
	print ("OT resolution:", t, "s")
	
comm.Barrier()
comm.Bcast([psi, mpi.DOUBLE],0)

Z,T_Z,psi_Z = eval_legendre_fenchel(mu, grad, psi)
interpol = make_cubic_interpolator(Z,T_Z,psi_Z,grad=grad)

##### Ray tracing #####
source_box = [np.min(Z[:,0]), np.max(Z[:,0]), np.min(Z[:,1]), np.max(Z[:,1])]
target_box = [np.min(Y[:,0]), np.max(Y[:,0]), np.min(Y[:,1]), np.max(Y[:,1])]
M = ray.ray_tracer(comm, s1, source_box, target_box, interpol, e_eta, e_ksi, n_plan, niter=3)
Mrecv = np.zeros((M.shape[0],M.shape[1]))

comm.Reduce([M,mpi.DOUBLE],[Mrecv,mpi.DOUBLE],op=mpi.SUM,root=0)

if rank == 0:
	
	Mrecv = 255.0*Mrecv/np.amax(Mrecv)
	plt.imshow(Mrecv, interpolation='nearest',
				   vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
	plt.show()
	print ("Execution time:", time.clock() - debut, "s")
"""
s = plt.scatter(X[:,0], X[:,1], c='b', marker=".", lw=0.1)
t = plt.scatter(gradx, grady, color='r', marker=".", lw=0.1)
#t1 = plt.scatter(gx1, gy1, color='g', marker=".", lw=0.1)
fig = plt.gcf()
ax = plt.gca()
ax.cla() # clear things for fresh plot
fig.gca().add_artist(s)
fig.gca().add_artist(t)
#fig.gca().add_artist(t1)
plt.show()"""
	
