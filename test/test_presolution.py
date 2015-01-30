from __future__ import print_function
import sys
sys.path.append('../lib')
import presolution as pre
import math
import numpy as np
import unittest

class PresolutionTest(unittest.TestCase):
	"""
	Test of presolution function by
	sending a square into a triangle.
	"""
	sPts = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.])
	sourceW = np.array([1.,2.,1.,2.])
	targetPts = np.array([[2.,0.],[3.,0.],[2.,1.])
	targetW = np.asfarray(1.,2.,1.)
	
	# Problème d'accès aux variables de presolution
	
