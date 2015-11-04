from __future__ import print_function

import sys
import argparse
import time

sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')
sys.path.append('./lib')

import numpy as np
import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import MongeAmpere as ma
from preprocessing import *
import geometry as geo
import rayTracing as ray
import misc
import export

debut = time.clock()

#### Parameters initialization ####
parser = argparse.ArgumentParser()
param = init_parameters(parser)
display_parameters(param)
# Target base
target_plane_base = [param['e_eta'],param['e_xi'],param['n_plan']] 
# Source rays direction
s1 = np.array([0.,0.,1.])

##### Source and target processing #####
mu, Y, nu = input_preprocessing(param)
#misc.plot_density(mu)

source_box = [np.min(mu.vertices[:,0]), np.max(mu.vertices[:,0]), np.min(mu.vertices[:,1]), np.max(mu.vertices[:,1])]
target_plane_box = [np.min(Y[:,0]), np.max(Y[:,0]), np.min(Y[:,1]), np.max(Y[:,1])]

gradx, grady = geo.planar_to_gradient(Y[:,0],Y[:,1],target_plane_base,s1)
grad = np.vstack([gradx,grady]).T

print('Number of diracs: ', len(nu))
t = time.clock() - debut
print ("Inputs processing:", t, "s")

##### Optimal Transport problem resolution #####
psi0 = ma.optimal_transport_presolve_2(grad, mu.vertices, Y_w=nu, X_w=mu.values)
psi = ma.optimal_transport_2(mu, grad, nu, w0=psi0, verbose=True)

t = time.clock() - t
print ("OT resolution:", t, "s")

Z,T_Z,psi_Z = misc.eval_legendre_fenchel(mu, grad, psi)
interpol = misc.make_cubic_interpolator(Z, T_Z, psi_Z, grad=grad)

##### Export of the scaled reflector in .off and .ioff files #####
points = np.array([Z[:,0],Z[:,1],psi_Z]).T
##export.export_improved_off('square_cameraman1e2.ioff', points, grad, T_Z)
#export.export_off('square_monge_1e3.off', points, T_Z)
export.export_off('square_monge_1e3_horiz.off', points, T_Z, rot=True, param=param)

##### Ray tracing #####
M = ray.ray_tracer(s1, mu, target_plane_box, interpol, target_plane_base, niter=4)

M = 255.0*M/np.amax(M)
print ("Ray tracing:", time.clock() - t, "s")
plt.imshow(M, interpolation='nearest',
		       vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
print ("Execution time:", time.clock() - debut, "s (CPU time)")
plt.show()
