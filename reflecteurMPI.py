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

debut = time.clock()


#### Parameters initialization ####
parser = argparse.ArgumentParser()
param = init_parameters(parser)
if rank == 0:
	display_parameters(param)
# Target base
target_base = [param['e_eta'],param['e_ksi'],param['n_plan']] 
#Source rays direction
s1 = np.array([0.,0.,1.])


##### Source and target processing #####
mu, Y, nu = input_preprocessing(param)
comm.Bcast([Y, mpi.DOUBLE],0)
comm.Bcast([nu, mpi.DOUBLE],0)
gradx, grady = geo.planar_to_gradient(Y[:,0], Y[:,1], target_base, s1)
grad = np.vstack([gradx,grady]).T


##### Optimal Transport problem resolution #####
psi = np.empty(len(gradx))
if rank == 0:
	print('Number of diracs: ', len(nu))
	t = time.clock() - debut
	print ("Inputs processing:", t, "s")

	psi = ma.optimal_transport_2(mu, grad, nu, verbose=True)
	
	t = time.clock() - t
	print ("OT resolution:", t, "s")
	
comm.Bcast([psi, mpi.DOUBLE],0)

Z,T_Z,psi_Z = misc.eval_legendre_fenchel(mu, grad, psi)
interpol = misc.make_cubic_interpolator(Z,T_Z,psi_Z,grad=grad)

source_box = [np.min(Z[:,0]), np.max(Z[:,0]), np.min(Z[:,1]), np.max(Z[:,1])]
target_box = [np.min(Y[:,0]), np.max(Y[:,0]), np.min(Y[:,1]), np.max(Y[:,1])]

	
##### Ray tracing #####
M = ray.ray_tracer(comm, s1, mu, target_box, interpol, target_base, niter=50)
Mrecv = np.zeros((M.shape[0],M.shape[1]))

comm.Reduce([M,mpi.DOUBLE],[Mrecv,mpi.DOUBLE],op=mpi.SUM,root=0)

if rank == 0:
	
	Mrecv = 255.0*Mrecv/np.amax(Mrecv)
	print ("Ray tracing:", time.clock() - t, "s")
	plt.imshow(Mrecv, interpolation='nearest',
				   vmin=0, vmax=255, cmap=plt.get_cmap('gray'))	
	print ("Execution time:", time.clock() - debut, "s (CPU time)")
	plt.show()
