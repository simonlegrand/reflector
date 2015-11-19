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

##### Source and target processing #####
mu, P, nu = input_preprocessing(param)
#misc.plot_density(mu)

source_box = [np.min(mu.vertices[:,0]), np.max(mu.vertices[:,0]), np.min(mu.vertices[:,1]), np.max(mu.vertices[:,1])]
target_plane_box = [np.min(P[:,0]), np.max(P[:,0]), np.min(P[:,1]), np.max(P[:,1])]

p,q = geo.planar_to_gradient(P[:,0],P[:,1],target_plane_base)
Y = np.vstack([p,q]).T

print('Number of diracs: ', len(nu))
t = time.clock() - debut
print ("Inputs processing:", t, "s")

##### Optimal Transport problem resolution #####
psi0 = ma.optimal_transport_presolve_2(Y, mu.vertices, Y_w=nu, X_w=mu.values)
psi = ma.optimal_transport_2(mu, Y, nu, w0=psi0, verbose=True)

t = time.clock() - t
print ("OT resolution:", t, "s")

Z,T_Z,u_Z = misc.eval_legendre_fenchel(mu, Y, psi)
interpol = misc.make_cubic_interpolator(Z, T_Z, u_Z, grad=Y)

##### Export of the scaled reflector in .off and .ioff files #####
points = np.array([Z[:,0],Z[:,1],u_Z]).T
##export.export_improved_off('square_cameraman1e2.ioff', points, grad, T_Z)
export.export_off('square_monge_1e3.off', points, T_Z)
#export.export_off('square_monge_1e3_horiz.off', points, T_Z, rot=True, param=param)

##### Ray tracing #####
M = ray.ray_tracer(mu, target_plane_box, interpol, target_plane_base, niter=4)

print ("Ray tracing:", time.clock() - t, "s")
plt.imshow(M, interpolation='nearest',
		       vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
print ("Execution time:", time.clock() - debut, "s (CPU time)")
plt.show()
