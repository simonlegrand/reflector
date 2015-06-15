#This script generates input files for the reflector program.
#The shapes are defined by the number of points and there
#coordinates.

import sys
sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')
import os
import numpy as np
import matplotlib.pyplot as plt
import MongeAmpere as ma

##### Regular polygon #####
if sys.argv[1] == "polygon":
	r = 0.5					# Radius of the polygon circumscribed circle 
	Nx = 200					# Number of vertices of the convex enveloppe.
	c = [0.0, 0.0]			# Center coordinates
	t = np.linspace(0,2*np.pi, Nx+1);
	t = t[0:Nx]
	shape = r * np.vstack([c[0]+np.cos(t), c[1]+np.sin(t)]).T;

if sys.argv[1] == "rectangle":
	Nx = 4
	w = 2.
	h = 1.
	xmin = 0.0
	ymin = 1.0
	shape = np.array([[xmin,ymin],[xmin+w,ymin],[xmin,ymin+h],[xmin+w,ymin+h]])
	
##### Optimized sampling of the shape #####
N = 100000
X = ma.optimized_sampling_2(ma.Density_2(shape), N, verbose=True);

##### Writing in the file #####
myFile = file('input.txt','w')

try:
	myFile.write('Reflector input file'+'\n')
	for i in range(0, N):
			myFile.write(str(X[i][0])+'\t'+str(X[i][1])+'\t'+str(1.0)+'\n')

finally:
	myFile.close()

poly = plt.scatter(X[:,0], X[:,1]  , color='b')

fig = plt.gcf()
ax = plt.gca()
ax.cla() # clear things for fresh plot

fig.gca().add_artist(poly)
plt.show()
