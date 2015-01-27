from geometry import *
import numpy as np
import MongeAmpere as ma
import matplotlib.pyplot as plt
import scipy.interpolate as ipol
import scipy.sparse as sparse
from mpl_toolkits.mplot3d import Axes3D
from pyhull.convex_hull import ConvexHull

def presolution(sourcePts, sourceW, targetPts, targetW):
	"""
	This function calculates psi0, a first estimation of psi.
	"""
	sourcePts = np.asfarray(sourcePts)
	sourceW = np.asfarray(sourceW)
	targetPts = np.asfarray(targetPts)
	targetW = np.asfarray(targetW)
	
	barySource = barycentre(sourcePts, sourceW)
	baryTarget = barycentre(targetPts, targetW)

	cirCircleSourceR = furthestPt(sourcePts, barySource)	# Source circum circle radius centered on barySource
	
	targetHull = ConvexHull(targetPts)
	p = targetHull.points
	v = targetHull.vertices
	
	dmin = distPtLine(p[v[0][0]], p[v[0][1]], baryTarget)
	for i in range(1, len(v)):
		if distPtLine(p[v[i][0]], p[v[i][1]], baryTarget) < dmin:
			dmin = distPtLine(p[v[i][0]], p[v[i][1]], baryTarget)
			
	insCircleTargetR = dmin									#Target inscribed circle radius centered on baryTarget
	
	ratio = insCircleTargetR / (cirCircleSourceR)
	
	gradx = ratio*(sourcePts[:,0]-barySource[0]) + baryTarget[0]	#Gradient must send source into target
	grady = ratio*(sourcePts[:,1]-barySource[1]) + baryTarget[1]
	
	psi_tilde0 = 0.5*ratio*(np.power(sourcePts[:,0] - barySource[0], 2) + np.power(sourcePts[:,1] - barySource[1], 2)) + baryTarget[0]*(sourcePts[:,0]) + baryTarget[1]*(sourcePts[:,1])

	"""fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(sourcePts[:,0], sourcePts[:,1], psi_tilde0)
	plt.show()"""
	psi0 = np.power(sourcePts[:,0], 2) + np.power(sourcePts[:,1], 2) - 2*psi_tilde0
	
	#Verifier que grad(psi_tilde0) est bien inclus dans targetPts
	"""circle1 = plt.Circle(barySource,cirCircleSourceR, color='b', fill=False)
	circle2 = plt.Circle(baryTarget,insCircleTargetR, color='r', fill=False)
	points1 = plt.scatter(sourcePts[:,0], sourcePts[:,1], color='b')
	points2 = plt.scatter(targetPts[:,0], targetPts[:,1], color='r')
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
	plt.show()"""
	return psi0
