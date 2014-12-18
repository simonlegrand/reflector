import sys
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image

hauteur = 128
largeur = 2*hauteur
img = Image.new("L",(largeur,hauteur))

for i in range(largeur):
	for j in range(hauteur):
		if (i<(largeur/8) and j<(hauteur/4)) or (i>(7*largeur/8) and j>(3*hauteur/4)):
			img.putpixel((i, j),20)
	
		else:
			img.putpixel((i, j),255)
print img
plt.imshow(img, vmin=0, vmax=255, cmap=plt.get_cmap('gray'));
plt.show()

img.save("outfile.png") 
#scipy.misc.imsave('outfile.jpg', img)
