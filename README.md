Reflector
====================
This program is designed to compute the shape of a reflector surface which transform a given beam of light
into a specified output intensity.

Dependencies
============
+ MongeAmpere/PyMongeAmpere
+ numpy 1.9
+ pyhull
+ pillow

Run the program
===============
``` sh
python reflecteur.py [source] [target]
```
If no argument is given, default is square source send on a triangle target.

Warning
=======
The algorithm used to solve the semi-discrete Monge-Ampere equation only works
for a convex density. If source or target is non-convex, you can use the
"XY" switch in the code to chose which of the source or target is a density.


