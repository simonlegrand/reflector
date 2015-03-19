from geometry import *
import numpy as np
import MongeAmpere as ma
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

def presolution(Y, mu, Y_w=None, X_w=None):
	"""
	This function calculates psi0, a first estimation of psi.
	
	Parameters
	----------
	Y : 2D array
		Target samples
	Y_w : 1D array
		Weights associated to Y
	X : 2D array
		Source samples
	X_w : 1D array
		Weights asociated to X
		
	Returns
	-------
	psi0 : 1D array
		Convex estimation of the potential. Its gradient
		send Y convex hull into X convex hull.	
	"""
	
	if X_w is None:
		X_w = np.ones(X.shape[0])
	if Y_w is None:
		Y_w = np.ones(Y.shape[0])
	
	bary_X = barycentre(X, X_w)
	bary_Y = barycentre(Y, Y_w)
	
	# Y circum circle radius centered on bary_Y
	r_Y = furthest_point(Y, bary_Y)

	X_hull = ConvexHull(X)
	points = X_hull.points
	simplices = X_hull.simplices
	
	# Search of the largest inscribed circle
	# centered on Y barycentre
	dmin = distance_point_line(points[simplices[0][0]], points[simplices[0][1]], bary_X)
	for simplex in simplices:
		d = distance_point_line(points[simplex[0]], points[simplex[1]], bary_X)
		if d < dmin:
			dmin = d
	# Y inscribed circle radius centered on bary_Y		
	r_X = dmin
	
	ratio = r_X / r_Y

	psi_tilde0 = 0.5*ratio*(np.power(Y[:,0] - bary_Y[0], 2) + np.power(Y[:,1] - bary_Y[1], 2)) + bary_X[0]*(Y[:,0]) + bary_X[1]*(Y[:,1])

	psi0 = np.power(Y[:,0], 2) + np.power(Y[:,1], 2) - 2*psi_tilde0
	"""
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(Y[:,0], Y[:,1], psi_tilde0)
	plt.show()"""
	
	#Verifier que grad(psi_tilde0) est bien inclus dans X
	
	gradx = ratio*(Y[:,0]-bary_Y[0]) + bary_X[0]
	grady = ratio*(Y[:,1]-bary_Y[1]) + bary_X[1]
	circle1 = plt.Circle(bary_X,r_X, color='b', fill=False)
	circle2 = plt.Circle(bary_Y,r_Y, color='r', fill=False)
	points1 = plt.scatter(Y[:,0], Y[:,1], color='b')
	points2 = plt.scatter(X[:,0], X[:,1], color='r')
	grad = plt.scatter(gradx, grady, color='g')
	fig = plt.gcf()
	ax = plt.gca()
	ax.cla() # clear things for fresh plot
	#ax.set_xlim((-0.5,5))
	#ax.set_ylim((-0.5,5))
	fig.gca().add_artist(circle1)
	fig.gca().add_artist(points1)
	fig.gca().add_artist(circle2)
	fig.gca().add_artist(points2)
	fig.gca().add_artist(grad)
	plt.show()
	
	return psi0
