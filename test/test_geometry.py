from __future__ import print_function
import sys
sys.path.append('../lib')
import geometry as geo
import math
import numpy as np
import unittest

class GeometryTest(unittest.TestCase):

	def setUp(self):
		"""DO NOT MODIFY!"""
		# Square
		self.p2 = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
		# Cube
		self.p3 = np.array([[0.,0.,0.],[0.,1.,0.],[1.,0.,0.],[1.,1.,0.],[0.,0.,1.],[0.,1.,1.],[1.,0.,1.],[1.,1.,1.]])

		
	def test_barycentre(self):
		w2 = np.array([1., 3., 1., 3.])
		self.assertTrue(np.allclose(geo.barycentre(self.p2, w2), [0.5, 0.75]))
		
		w3 = np.array([1., 3., 1., 3., 3., 1., 3., 1.])
		self.assertTrue(np.allclose(geo.barycentre(self.p3, w3), [0.5, 0.5, 0.5]))
		
		self.assertRaises(geo.NotProperShapeError, geo.barycentre(self.p3, w2))
		
		
		
	def test_furthestPt(self):
		a2 = np.array([-1., -1.])
		self.assertTrue(np.allclose(geo.furthestPt(self.p2, a2), [2.*math.sqrt(2), 2.*math.sqrt(2)]))
		
		a3 = np.array([-1., -1., -1.])
		self.assertTrue(np.allclose(geo.furthestPt(self.p3, a3), [2.*math.sqrt(3), 2.*math.sqrt(3)]))
		
		self.assertRaises(geo.NotProperShapeError, geo.furthestPt(self.p3, [[0.,0.,0.], [0.,0.,0.]]))
			
		self.assertRaises(geo.NotProperShapeError, geo.furthestPt(self.p3, [0.,0.]))
			
		
	def test_distPtLine(self):
		self.assertTrue(np.allclose(geo.distPtLine([0.,0.], [0.,1.], [1.,1.]), 1.))
		self.assertTrue(np.allclose(geo.distPtLine([0.,0.,0.], [0.,0.,1.], [1.,1.,0.]), math.sqrt(2)))
		self.assertTrue(np.allclose(geo.distPtLine([0.,0.,0.], [1.,1.,1.], [1.,1.,0.]), math.sqrt(6)/3.))
		
		self.assertRaises(geo.NotProperShapeError, geo.distPtLine([0.,0.], [0.,1.], [1.,1.,0.]))
		a = np.array([[0.,3.],
					[3.,0.],
					[2.,1.]])

		self.assertRaises(geo.NotProperShapeError, geo.distPtLine([0.,0.], [0.,1.], a))
		self.assertRaises(geo.NotProperShapeError, geo.distPtLine([0.,0.], [0.,0.], [1.,1.]))
