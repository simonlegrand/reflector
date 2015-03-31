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

debut = time.clock()

parser = argparse.ArgumentParser()

#### Source rays direction ####
s1 = np.array([0.,0.,1.])

#### Target plan basis ####
n_plan = np.array([-5.*np.cos(np.pi/4),0.,5.*np.sin(np.pi/4)])
e_eta = np.array([0.,1.,0.])
e_ksi = np.array([np.sin(np.pi/4),0.,1.*np.cos(np.pi/4)])
target_base = [e_eta,e_ksi,n_plan] 

##### Source and target processing #####
mu, Y, nu = input_preprocessing(parser)
misc.plot_density(mu)

gradx, grady = geo.planar_to_gradient(Y[:,0],Y[:,1],s1,target_base)
grad = np.vstack([gradx,grady]).T

print('Number of diracs: ', len(nu))
t = time.clock() - debut
print ("Inputs processing:", t, "s")

##### Optimal Transport problem resolution #####
psi = ma.optimal_transport_2(mu, grad, nu, verbose=True)
#misc.write_data(psi, 'refl.dat')
#psi = misc.load_data('../testHeliospectra/uniformSource/refl.dat')
t = time.clock() - t
print ("OT resolution:", t, "s")

Z,T_Z,psi_Z = eval_legendre_fenchel(mu, grad, psi)
interpol = make_cubic_interpolator(Z, T_Z, psi_Z, grad=grad)

source_box = [np.min(Z[:,0]), np.max(Z[:,0]), np.min(Z[:,1]), np.max(Z[:,1])]
target_box = [np.min(Y[:,0]), np.max(Y[:,0]), np.min(Y[:,1]), np.max(Y[:,1])]
misc.plot_reflector(interpol,source_box)

##### Ray tracing #####
M = ray.ray_tracer(s1, mu, target_box, interpol, target_base, niter=5)

M = 255.0*M/np.amax(M)
plt.imshow(M, interpolation='nearest',
		       vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
print ("Execution time:", time.clock() - debut, "s")
plt.show()
