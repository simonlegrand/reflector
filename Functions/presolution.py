import sys
from geometry2D import *
import numpy as np
import MongeAmpere as ma
import matplotlib.pyplot as plt
import scipy.interpolate as ipol
import scipy.sparse as sparse
from mpl_toolkits.mplot3d import Axes3D
from pyhull.convex_hull import ConvexHull

def presolution(sourcePts, sourceW, targetPts, targetW):
	"""This function alculates psi0, a first estimation of psi."""
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
	translation = baryTarget - barySource
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
	
"""ltarget = 1.0
xmin = 3.0
ymin = 2.0
X = np.array([[xmin,ymin],[xmin+ltarget,ymin],[xmin,ymin+ltarget],[xmin+ltarget,ymin+ltarget]])
mu = np.array([1., 1., 1., 1.])
mumoy = np.sum(mu)/len(mu)
dens = ma.Density_2(X, mu)
Nx = 30;
t = np.linspace(0,2*np.pi,Nx+1);
t = t[0:Nx]
disk = np.vstack([np.cos(t),np.sin(t)]).T;
X = ma.Density_2(disk).optimized_sampling(1000,verbose=True);
mu = np.ones(1000)
dens = ma.Density_2(X,mu);

Lsource = 2.0
N = 100
Ndirac = N*N
squareSource = np.array([[0.,0.],[Lsource,0.],[0.,Lsource],[Lsource,Lsource]])
weightsSource = np.array([1., 1., 1., 1.])
Y = ma.Density_2(squareSource).optimized_sampling(Ndirac-4)
Y = np.concatenate((Y, squareSource))
nu = (dens.mass()/Ndirac) * np.ones(Ndirac)

psi = ma.optimal_transport_2(dens, Y, nu, presolution(Y, nu, X, mu), verbose=True)
#psi_tilde0 = presolution(Y, nu, squareTarget, weightsTarget)
psi_tilde = (Y[:,1]*Y[:,1] + Y[:,0]*Y[:,0] - psi)/2
interpol = ipol.CloughTocher2DInterpolator(Y, psi_tilde, tol=1e-6)


##### Gradient calculation #####
nmesh = 10*N
[x,y] = np.meshgrid(np.linspace(0., Lsource, nmesh),
                            np.linspace(0., Lsource, nmesh))

Nx = nmesh*nmesh

x = np.reshape(x,(Nx))
y = np.reshape(y,(Nx))
X = np.vstack([x,y]).T								# Cree une matrice (Nx,2) de coordonnees des noeuds de la grille
source = plt.scatter(X[:,0], X[:,1]  , color='g')
I=np.reshape(interpol(X),(nmesh,nmesh))
[gy, gx] = np.gradient(I, Lsource/nmesh, Lsource/nmesh)

gx = np.reshape(gx, (nmesh*nmesh))			# Remise sous forme de vecteur pour chaque coordonnee du gradient
gy = np.reshape(gy, (nmesh*nmesh))

threshold = np.ones(Nx)
I = np.logical_or(np.logical_or(np.greater(gx, threshold*(xmin+ltarget)), np.greater(gy, threshold*(ymin+ltarget))), np.logical_or(np.less(gx, threshold*xmin), np.less(gy, threshold*ymin)))
print I
X = X[I]
J = np.logical_and(np.less(gx, threshold*(xmin+ltarget)), np.less(gy, threshold*(ymin+ltarget)))
gx = gx[J]
gy = gy[J]
J = np.logical_and(np.greater(gx, np.ones(gx.size)*xmin), np.greater(gy, np.ones(gy.size)*ymin))
gx = gx[J]
gy = gy[J]
J = np.logical_and(np.isfinite(gx) ,np.isfinite(gy))
gx = gx[J]
gy = gy[J]

out = plt.scatter(X[:,0], X[:,1]  , color='b')
grad = plt.scatter(gx, gy, color='r')
fig = plt.gcf()
ax = plt.gca()
ax.cla() # clear things for fresh plot
fig.gca().add_artist(source)
fig.gca().add_artist(grad)
fig.gca().add_artist(out)
plt.show()
"""
