#This script generates input files for the reflector program.
#The shapes are defined by the number of points and there
#coordinates.

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

##### Regular polygon #####
if sys.argv[1] == "polygon":
	r = 0.5					# Radius of the polygon circumscribed circle 
	Nx = 50					# Number of vertices
	c = [0.0, 0.0]			# Center
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
##### Writing in the file #####
myFile = file('input.txt','w')

try:
	myFile.write('Input file reflector'+'\n')
	for i in range(0, Nx):
			myFile.write(str(shape[i][0])+'\t'+str(shape[i][1])+'\n')

finally:
	myFile.close()

poly = plt.scatter(shape[:,0], shape[:,1]  , color='b')

fig = plt.gcf()
ax = plt.gca()
ax.cla() # clear things for fresh plot

fig.gca().add_artist(poly)
plt.show()
