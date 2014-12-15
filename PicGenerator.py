import sys
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

n = 128
img = np.ones((n,n))
dims = np.shape(img)
for i in range(dims[0]):
	for j in range(dims[1]):
		if i<(n/4) and j<(n/4):
			img[i][j] = 255
scipy.misc.imsave('outfile.jpg', img)
