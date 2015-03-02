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
+ mpi4py (optionnal)

Run the program
===============
``` sh
python reflecteur.py [source] [target]
```
If no argument is given, default is square source send on a triangle target.

Warning
=======
The algorithm used to solve the semi-discrete Monge-Ampere equation only works
for a convex density.

Parallelized version
====================
reflecteurMPI.py is a parallelized version of reflecteur.py. The ray tracing is the only function prallelized, it is useless to use it if you are not concerned by ray tracing. The command to run the program is:
``` sh
mpiexec -n <nbofprocess> python reflecteur.py [source] [target]
```



