# PyMongeAmpere
# Copyright (C) 2014 Quentin Merigot, CNRS
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import sys
sys.path.append('../PyMongeAmpere-build/');
sys.path.append('../PyMongeAmpere-build/lib');
import os
import MongeAmpere as ma
import numpy as np
import scipy as sp
import scipy.interpolate as ipol
import scipy.optimize as opt
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
import time

debut = time.clock();

fig = plt.figure();

def presolution(img, res, nbdirac, grid):
	n0 = int(np.floor(res/2));                             # la triangulation contient n^2 points
	mu0 = sp.misc.imresize(img, (n0,n0));			# Redimensionnement de l'image en n0*n0
	mu0 = np.asarray(mu0, dtype=float);			# Inverse les niveau de gris et transforme en matrice
	dens0 = ma.Density_2.from_image(mu0,[0,1,0,1]); 	# dens = cible, objet de type Density_2
	
	nu0 = (dens0.mass()/nbdirac) * np.ones(nbdirac);
	return ma.optimal_transport_2(dens0, Y, nu0, verbose=True);

# cible (du probleme du reflecteur) = source pour le transport optimal = densite
cloudname = "lena.cloud";
img = sp.misc.lena();

n = 128;                            # la triangulation contient n^2 points
mu = sp.misc.imresize(img, (n,n));  # Redimensionnement de l'image en n0*n0
mu = np.asarray(mu, dtype=float);   # Transforme en matrice de float

mumoy = np.sum(mu)/(n*n);	    # Calcul de la valeur moyenne des pixels

dens = ma.Density_2.from_image(mu,[0,1,0,1]); # dens = cible, objet de type Density_2, contient les points de la triangularisation et leur densite associee.

# source (du probleme du reflecteur) = target pour le transport optimal = Diracs sur le carre
N=100;
square = np.array([[0.,0.],[0.,1.],[1.,1.],[1.,0.]]);

# Echantillonage regulier #######
[x,y] = np.meshgrid(np.linspace(0.,1.,N),
                           np.linspace(0.,1.,N)); # Retourne deux matrices : Matrices coordonnees de x et de y

Ny = N*N;

x = np.reshape(x,(Ny));
y = np.reshape(y,(Ny));
Y = np.vstack([x,y]).T; # .T=transpose
#################################

# Echantillonage adapte #########
#Y = ma.Density_2(square).optimized_sampling(N*N); # Utile si la source n'est pas uniforme

nu = (dens.mass()/Ny) * np.ones(Ny); # Normalisation

# resolution du probleme de TO (avec ou sans presolution pour psi)
#psi = ma.optimal_transport_2(dens, Y, nu, verbose=True);
psi = ma.optimal_transport_2(dens, Y, nu, presolution(img, n, Ny, Y), verbose=True);

[Z,m] = ma.lloyd_2(dens,Y,psi); # Z = centroides du diagramme de Laguerre (Y,psi); m = masse des cellules
#plt.scatter(Z[:,0],Z[:,1],s=1);
#plt.show()


# Trouver le psi tilde (verifier dans Cgal l'expression des cellules de laguerre)
psi_tilde = (Y[:,0]*Y[:,0] + Y[:,1]*Y[:,1] - psi)/2;

# Tracer f= (x, y, psi tilde) et verifier que c'est convexe.
#ax = Axes3D(fig)
#ax.scatter(Y[:,0], Y[:,1], psi_tilde);
#plt.show()

interpol = ipol.CloughTocher2DInterpolator(Y, psi_tilde, tol=1e-6); # Interpolation de psi
#print interpol(Y);

##### Creation d'une grille X grace a laquelle on va evaluer le gradient de interpol
nmesh = 30*N;
[x,y] = np.meshgrid(np.linspace(0.,1.,nmesh),
                            np.linspace(0.,1.,nmesh)); # Retourne deux matrices : Matrices coordonnees de x et de y

Nx = nmesh*nmesh;

x = np.reshape(x,(Nx));
y = np.reshape(y,(Nx));
X = np.vstack([x,y]).T;				# Cree une matrice (Nx,2) de coordonnees des noeuds de la grille
I=np.reshape(interpol(X),(nmesh,nmesh));	# Valeurs de psi_tilde aux noeuds de la grille

[gx, gy] = np.gradient(I, 1./nmesh, 1./nmesh);	# On evalue le gradient

gx = np.reshape(gx, (nmesh*nmesh));		# Remise sous forme de vecteur pour chaque coordonnee du gradient
gy = np.reshape(gy, (nmesh*nmesh));

# Pseudo lancer de rayon
# (gx,gy) = "position des rayons reflechis"
# on en deduit des coordonnees de pixels (ix,iy) dans une image de taille npix * npix
npix = N;
ix = np.floor(npix*gx); 			# Coordonnees x des pixels ou vont tomber les photons de coord gx
iy = np.floor(npix*gy);				# Coordonnees y	""
data = np.ones(nmesh*nmesh);
M = sparse.coo_matrix((data, (ix,iy)), shape=(npix,npix)).todense();

Mmoy = np.sum(M)/(npix*npix);
M = M/(Mmoy/mumoy);				# Egalisation de la valeur moyenne des pixels


print "Execution time:", time.clock() - debut;

plt.imshow(M, interpolation='nearest', vmin=0, vmax=255, cmap=plt.get_cmap('gray'));
#plt.scatter(gx, gy);
plt.show()
