#!/usr/bin/python
from __future__ import print_function

import sys
import argparse
import time

sys.path.append('./lib')
sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')

import numpy as np
import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpi4py.MPI as mpi

import MongeAmpere as ma
from preprocessing import *
import geometry as geo
import rayTracingMPI as ray
import misc

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

parser = argparse.ArgumentParser()

##### Source and target processing #####
X, mu, Y, nu = input_preprocessing(parser)
gradx, grady = geo.planar_to_gradient(Y[:,0], Y[:,1], e_eta=e_eta, e_ksi=e_ksi, n=n_plan)
grad = np.vstack([gradx,grady]).T
psi = np.empty(len(gradx))

if rank == 0:
	t = time.clock() - debut
	print ("Inputs processing:", t, "s")

	##### Optimal Transport problem resolution #####
	psi = ma.optimal_transport_2(mu, grad, nu, verbose=True, X=X)
	#misc.write_data(psi, 'refl.dat')
	#psi = misc.load_data('../testHeliospectra/uniformSource/refl.dat')
	
	t = time.clock() - t
	print ("OT resolution:", t, "s")
	
comm.Barrier()
comm.Bcast([psi, mpi.DOUBLE],0)

Z,T_Z,psi_Z = eval_legendre_fenchel(mu, grad, psi)
interpol = make_cubic_interpolator(Z,T_Z,psi_Z,grad=grad)

source_box = [np.min(Z[:,0]), np.max(Z[:,0]), np.min(Z[:,1]), np.max(Z[:,1])]
target_box = [np.min(Y[:,0]), np.max(Y[:,0]), np.min(Y[:,1]), np.max(Y[:,1])]
#if rank == 0:
#	misc.plot_reflector(interpol,source_box)
	
##### Ray tracing #####
M = ray.ray_tracer(comm, s1, mu, source_box, target_box, interpol, e_eta, e_ksi, n_plan, niter=4)
Mrecv = np.zeros((M.shape[0],M.shape[1]))

comm.Reduce([M,mpi.DOUBLE],[Mrecv,mpi.DOUBLE],op=mpi.SUM,root=0)

if rank == 0:
	
	Mrecv = 255.0*Mrecv/np.amax(Mrecv)
	plt.imshow(Mrecv, interpolation='nearest',
				   vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
	print ("Execution time:", time.clock() - debut, "s")
	plt.show()
