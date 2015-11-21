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
import export

comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

debut = time.clock()


#### Parameters initialization ####
parser = argparse.ArgumentParser()
param = init_parameters(parser)
if rank == 0:
	display_parameters(param)
target_base = [param['e_eta'],param['e_xi'],param['n_plan']] 


##### Source and target processing #####
mu, P, nu = input_preprocessing(param)
comm.Bcast([P, mpi.DOUBLE],0)
comm.Bcast([nu, mpi.DOUBLE],0)
p,q = geo.planar_to_gradient(P[:,0], P[:,1], target_base)
Y = np.vstack([p,q]).T


##### Optimal Transport problem resolution #####
psi = np.empty(len(p))
if rank == 0:
	print('Number of diracs: ', len(nu))
	t = time.clock() - debut
	print ("Inputs processing:", t, "s")

	psi0 = ma.optimal_transport_presolve_2(Y, mu.vertices, Y_w=nu, X_w=mu.values)
	psi = ma.optimal_transport_2(mu, Y, nu, w0=psi0, verbose=True)
	
	t = time.clock() - t
	print ("OT resolution:", t, "s")
	
comm.Bcast([psi, mpi.DOUBLE],0)

Z,T_Z,u_Z = misc.eval_legendre_fenchel(mu, Y, psi)
interpol = misc.make_cubic_interpolator(Z,T_Z,u_Z,grad=Y)

source_box = [np.min(Z[:,0]), np.max(Z[:,0]), np.min(Z[:,1]), np.max(Z[:,1])]
target_box = [np.min(P[:,0]), np.max(P[:,0]), np.min(P[:,1]), np.max(P[:,1])]

	
##### Ray tracing #####
<<<<<<< HEAD
M = ray.ray_tracer(comm, mu, target_box, interpol, target_base, niter=4)
=======
M = ray.ray_tracer(comm, mu, target_box, interpol, target_base, niter=6)
>>>>>>> 6c522110e1d74b3bc965d777e8e259c4a2196b7b
Mrecv = np.zeros((M.shape[0],M.shape[1]))

comm.Reduce([M,mpi.DOUBLE],[Mrecv,mpi.DOUBLE],op=mpi.SUM,root=0)

if rank == 0:
	
	Mrecv = 255.0*Mrecv/np.amax(Mrecv)
	print ("Ray tracing:", time.clock() - t, "s")
	plt.imshow(Mrecv, interpolation='nearest',
				   vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
				   
	##### Export of the scaled reflector in .off and .ioff files #####
	points = np.array([Z[:,0],Z[:,1],u_Z]).T
	##export.export_improved_off('square_cameraman1e2.ioff', points, grad, T_Z)
	#export.export_off('square_monge_1e3.off', points, T_Z)
	export.export_off('square_monge_horiz.off', points, T_Z, rot=True, param=param)
	
	print ("Execution time:", time.clock() - debut, "s (CPU time)")
	plt.show()
