Reflector
====================
This program is designed to compute the shape of a reflector surface which transform a given beam of light
into a specified output intensity.

Dependencies
============
+ MongeAmpere/PyMongeAmpere
+ numpy 1.9
+ matplotlib 1.4
+ pillow
+ mpi4py (optionnal)

Run the program
===============
``` sh
reflecteur.py [-h] [--s s] [--t t]

optional arguments:
  -h, --help         show this help message and exit
  --s s, --source s  source file name
  --t t, --target t  target file name

```
Default source is a uniform square source and
default target is a uniform triangle with 10000 diracs.

Warning
=======
The algorithm used to solve the semi-discrete Monge-Ampere equation only works
for a convex density.

Parallelized version
====================
reflecteurMPI.py is a parallelized version of reflecteur.py. The ray tracing is the only function prallelized, it is useless to use it if you are not concerned by ray tracing. The command to run the program is:
``` sh
mpiexec -n <nbofprocess> python reflecteur.py [-h] [--s s] [--t t]
```



