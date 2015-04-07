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
python reflecteur.py [-h] [--f f]
```
optional arguments:
``` sh
-h, --help         show this help message and exit
--f f, --file f	   parameter file
```

Default source is a uniform square and
default target is a uniform triangle with 10000 diracs.

Warning
=======
The algorithm used to solve the semi-discrete Monge-Ampere equation only works
for a convex density.

Parallel version
====================
reflecteurMPI.py is a parallelized version of reflecteur.py. The ray tracing is the only function parallelized, it is useless to run it if you are not concerned by the resimulation of the reflector. The command to run the program is:
``` sh
mpirun -n <nbofprocess> python reflecteur.py [-h] [--f f]
```



