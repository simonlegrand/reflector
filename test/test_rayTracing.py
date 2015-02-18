from __future__ import print_function
import sys
sys.path.append('../lib')
import rayTracing as ray
import math
import numpy as np
import unittest

class RayTracingTest(unittest.TestCase):

	def setUp(self):
		self.s1 = np.array([[0.,0.,1.],[0.,1.,0.]])
		self.gradx = np.array([0.,1.])
		self.grady = np.array([1.,0.])
		self.grad = np.vstack([self.gradx, self.grady]).T
		
	def test_reflection(self):
	
		"""rint(self.s1)
		print(self.s1.shape)"""
		print(ray.reflection(self.grad,self.s1))
		#self.assertTrue(np.allclose(ray.reflection(self.s1,self.grad),np.array([[0.,1.,0.],[1.,0.5,0.]])))
		
