"""This script generates test shapes for the reflector application.
One among the three following arguments can be added on command line"""
import sys
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image

hauteur = 128
largeur = 2*hauteur
img = Image.new("L",(largeur,hauteur))

arg = sys.argv[1];

"""Generate a picture with squares at top left and bottom right"""
if arg == "corners":

	for i in range(largeur):
		for j in range(hauteur):
			if (i<(largeur/8) and j<(hauteur/4)) or (i>(7*largeur/8) and j>(3*hauteur/4)):
				img.putpixel((i, j),0)
	
			else:
				img.putpixel((i, j),255)

"""Generate a picture with a centered square"""
if arg == "center":

	rectangleh = hauteur/4.0		# Centered shape dimensions
	rectanglew = hauteur/4.0
	for i in range(largeur):
		for j in range(hauteur):
			if (i>(largeur/2.0-rectanglew/2.0) and i<(largeur/2.0+rectanglew/2.0)) and (j>(hauteur/2.0-rectangleh/2.0) and j<(hauteur/2.0+rectangleh/2.0)):
				img.putpixel((i, j),0)
	
			else:
				img.putpixel((i, j),255)

"""Generate a picture with a combination of previous cases"""
if arg == "both":

	rectangleh = hauteur/4.0		# Centered shape dimensions
	rectanglew = hauteur/4.0
	for i in range(largeur):
		for j in range(hauteur):
			if (i<(largeur/8) and j<(hauteur/4)) or (i>(7*largeur/8) and j>(3*hauteur/4)) or ((i>(largeur/2.0-rectanglew/2.0) and i<(largeur/2.0+rectanglew/2.0)) and (j>(hauteur/2.0-rectangleh/2.0) and j<(hauteur/2.0+rectangleh/2.0))):
				img.putpixel((i, j),0)
	
			else:
				img.putpixel((i, j),255)

print img
plt.imshow(img, vmin=0, vmax=255, cmap=plt.get_cmap('gray'));
plt.show()

img.save("outfile.png") 
#scipy.misc.imsave('outfile.jpg', img)
