from geometry import *
import numpy as np
import MongeAmpere as ma
import matplotlib.pyplot as plt
import scipy.interpolate as ipol
import scipy.sparse as sparse
from mpl_toolkits.mplot3d import Axes3D
from pyhull.convex_hull import ConvexHull

def presolution(target_pts, source_pts, source_w=None, target_w=None):
	"""
	This function calculates psi0, a first estimation of psi.
	
	Parameters
	----------
	target_pts : 2D array
	source_w : 1D array
		Weights associated to target_pts
	source_pts : 2D array
	target_w : 1D array
		Weights asociated to source_pts
		
	Returns
	-------
	psi0 : 1D array
		Convex estimation of the potential. Its gradient
		send target_pts convex hull into source_pts
		convex hull.	
	"""
	
	if source_w is None:
		source_w = np.ones(target_pts.shape[0])
	if target_w is None:
		target_w = np.ones(source_pts.shape[0])

	target_pts = np.asarray(target_pts, dtype=np.float64)
	source_w = np.asarray(source_w, dtype=np.float64)
	source_pts = np.asarray(source_pts, dtype=np.float64)
	target_w = np.asarray(target_w, dtype=np.float64)
	
	bary_source = barycentre(target_pts, source_w)
	bary_target = barycentre(source_pts, target_w)
	
	# Source circum circle radius centered on bary_source
	r_source = furthest_point(target_pts, bary_source)

	target_hull = ConvexHull(source_pts)
	p = target_hull.points
	v = target_hull.vertices
	
	dmin = distance_point_line(p[v[0][0]], p[v[0][1]], bary_target)
	for i in range(1, len(v)):
		if distance_point_line(p[v[i][0]], p[v[i][1]], bary_target) < dmin:
			dmin = distance_point_line(p[v[i][0]], p[v[i][1]], bary_target)
	
	# Target inscribed circle radius centered on bary_target		
	r_target = dmin
	
	ratio = r_target / r_source
	
	psi_tilde0 = 0.5*ratio*(np.power(target_pts[:,0] - bary_source[0], 2) + np.power(target_pts[:,1] - bary_source[1], 2)) + bary_target[0]*(target_pts[:,0]) + bary_target[1]*(target_pts[:,1])

	psi0 = np.power(target_pts[:,0], 2) + np.power(target_pts[:,1], 2) - 2*psi_tilde0
	"""
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(target_pts[:,0], target_pts[:,1], psi_tilde0)
	plt.show()"""
	
	#Verifier que grad(psi_tilde0) est bien inclus dans source_pts
	"""
	gradx = ratio*(target_pts[:,0]-bary_source[0]) + bary_target[0]
	grady = ratio*(target_pts[:,1]-bary_source[1]) + bary_target[1]
	circle1 = plt.Circle(bary_source,r_source, color='b', fill=False)
	circle2 = plt.Circle(bary_target,r_target, color='r', fill=False)
	points1 = plt.scatter(target_pts[:,0], target_pts[:,1], color='b')
	points2 = plt.scatter(source_pts[:,0], source_pts[:,1], color='r')
	grad = plt.scatter(gradx, grady, color='g')
	fig = plt.gcf()
	ax = plt.gca()
	ax.cla() # clear things for fresh plot
	ax.set_xlim((-0.5,5))
	ax.set_ylim((-0.5,5))
	fig.gca().add_artist(circle1)
	fig.gca().add_artist(points1)
	fig.gca().add_artist(circle2)
	fig.gca().add_artist(points2)
	fig.gca().add_artist(grad)
	plt.show()
	"""
	return psi0
