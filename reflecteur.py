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

# Target plan basis
n_plan = np.array([-5.*np.cos(np.pi/4),0.,5.*np.sin(np.pi/4)])
e_eta = np.array([0.,1.,0.])
e_ksi = np.array([np.sin(np.pi/4),0.,1.*np.cos(np.pi/4)])

##### Source and target processing #####
X, mu, Y, nu = input_preprocessing(parser)

gradx, grady = geo.planar_to_gradient(Y[:,0],Y[:,1],e_eta=e_eta,e_ksi=e_ksi,n=n_plan)
grad = np.vstack([gradx,grady]).T

t = time.clock() - debut
print ("Inputs processing:", t, "s")

##### Optimal Transport problem resolution #####
psi = ma.optimal_transport_2(mu, grad, nu, verbose=True, X=X)
#misc.write_data(psi, 'refl.dat')
#psi = misc.load_data('../testHeliospectra/uniformSource/refl.dat')
Z,T_Z,psi_Z = eval_legendre_fenchel(mu, grad, psi)
interpol = make_cubic_interpolator(Z, T_Z, psi_Z, grad=grad)

t = time.clock() - t
print ("OT resolution:", t, "s")

##### Ray tracing #####
s1 = np.array([0.,0.,1.])
source_box = [np.min(Z[:,0]), np.max(Z[:,0]), np.min(Z[:,1]), np.max(Z[:,1])]
target_box = [np.min(Y[:,0]), np.max(Y[:,0]), np.min(Y[:,1]), np.max(Y[:,1])]
M = ray.ray_tracer(s1, mu,  source_box, target_box, interpol, e_eta, e_ksi, n_plan, niter=4)

M = 255.0*M/np.amax(M)
plt.imshow(M, interpolation='nearest',
		       vmin=0, vmax=255, cmap=plt.get_cmap('gray'))
print ("Execution time:", time.clock() - debut, "s")
plt.show()
