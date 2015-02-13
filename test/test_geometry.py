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
		w2 = np.array([1.,3.,1.,3.])
		self.assertTrue(np.allclose(geo.barycentre(self.p2, w2),[0.5,0.75]))
		
		w3 = np.array([1.,3.,1.,3.,3.,1.,3.,1.])
		self.assertTrue(np.allclose(geo.barycentre(self.p3, w3),[0.5,0.5,0.5]))
		
		self.assertRaises(geo.NotProperShapeError, geo.barycentre(self.p3,w2))
		
		
	def test_furthest_point(self):
		a2 = np.array([-1.,-1.])
		self.assertTrue(np.allclose(geo.furthest_point(self.p2, a2),[2.*math.sqrt(2),2.*math.sqrt(2)]))
		
		a3 = np.array([-1.,-1.,-1.])
		self.assertTrue(np.allclose(geo.furthest_point(self.p3, a3),[2.*math.sqrt(3),2.*math.sqrt(3)]))
		
		self.assertRaises(geo.NotProperShapeError,geo.furthest_point(self.p3,[[0.,0.,0.],[0.,0.,0.]]))
			
		self.assertRaises(geo.NotProperShapeError,geo.furthest_point(self.p3,[0.,0.]))
			
		
	def test_distance_point_line(self):
		self.assertTrue(np.allclose(geo.distance_point_line([0.,0.],[0.,1.],[1.,1.]),1.))
		self.assertTrue(np.allclose(geo.distance_point_line([0.,0.,0.],[0.,0.,1.],[1.,1.,0.]),math.sqrt(2)))
		self.assertTrue(np.allclose(geo.distance_point_line([0.,0.,0.],[1.,1.,1.],[1.,1.,0.]),math.sqrt(6)/3.))
		
		self.assertRaises(geo.NotProperShapeError, geo.distance_point_line([0.,0.],[0.,1.],[1.,1.,0.]))
		a = np.array([[0.,3.],
					[3.,0.],
					[2.,1.]])

		self.assertRaises(geo.NotProperShapeError,geo.distance_point_line([0.,0.],[0.,1.],a))
		self.assertRaises(geo.NotProperShapeError,geo.distance_point_line([0.,0.],[0.,0.],[1.,1.]))
		
		
	def test_gradient_to_spherical(self):
		gradx = np.array([1.,1.])
		grady = np.array([1.,1.])
		self.assertTrue(np.allclose(geo.gradient_to_spherical(gradx,grady)[0],[np.arccos(1/3.),np.arccos(1/3.)]))
		self.assertTrue(np.allclose(geo.gradient_to_spherical(gradx,grady)[1],[np.pi/4.,np.pi/4.]))
		
		gradx = np.array([0.,1.])
		grady = np.array([1.,1.])
		self.assertRaises(FloatingPointError,geo.gradient_to_spherical(gradx,grady))
		
		gradx = np.array([[1.,3.],
					[3.,0.],
					[2.,1.]])
		self.assertRaises(geo.NotProperShapeError,geo.gradient_to_spherical(gradx,grady))
		
		gradx = np.array([1.,1.,1.])
		self.assertRaises(geo.NotProperShapeError,geo.gradient_to_spherical(gradx,grady))
		
		
	def test_spherical_to_gradient(self):
		theta = np.arccos(1/3.)
		phi = np.pi/4.
		[gradx, grady] = geo.spherical_to_gradient(theta, phi)
		self.assertTrue(np.allclose(geo.gradient_to_spherical(gradx,grady)[0],theta))
		self.assertTrue(np.allclose(geo.gradient_to_spherical(gradx,grady)[1],phi))
		
		theta = np.array([[1.,3.],
					[3.,0.],
					[2.,1.]])
		self.assertRaises(geo.NotProperShapeError,geo.spherical_to_gradient(theta,phi))
		
		theta = np.array([1.,1.,1.])
		self.assertRaises(geo.NotProperShapeError,geo.spherical_to_gradient(theta,phi))
		
	
	def test_planar_to_spherical(self):
		d = 1.
		theta_0 = np.pi/2
		phi_0 = 0
		self.assertTrue(np.allclose(geo.planar_to_spherical(0.,0.,theta_0,phi_0,d)[0],1))
		self.assertTrue(np.allclose(geo.planar_to_spherical(0.,0.,theta_0,phi_0,d)[1],theta_0))
		self.assertTrue(np.allclose(geo.planar_to_spherical(0.,0.,theta_0,phi_0,d)[2],phi_0))
		
		self.assertTrue(np.allclose(geo.planar_to_spherical(1.,1.,theta_0,phi_0,d)[0],np.sqrt(3)))
		self.assertTrue(np.allclose(geo.planar_to_spherical(1.,1.,theta_0,phi_0,d)[1],0.9553166))
		self.assertTrue(np.allclose(geo.planar_to_spherical(1.,1.,theta_0,phi_0,d)[2],-np.pi/4))
		
		phi_0 = np.pi
		self.assertTrue(np.allclose(geo.planar_to_spherical(1.,1.,theta_0,phi_0,d)[0],np.sqrt(3)))
		self.assertTrue(np.allclose(geo.planar_to_spherical(1.,1.,theta_0,phi_0,d)[1],0.9553166))
		self.assertTrue(np.allclose(geo.planar_to_spherical(1.,1.,theta_0,phi_0,d)[2],3*np.pi/4))
		
		theta_0 = 0
		phi_0 = np.pi
		self.assertTrue(np.allclose(geo.planar_to_spherical(1.,0,theta_0,phi_0,d)[2],np.pi/2))
		
		
	def test_spherical_to_cartesian(self):
		r = 1.
		theta = np.pi/2
		phi = 0
		self.assertTrue(np.allclose(geo.spherical_to_cartesian(r,theta,phi),[1.,0.,0.]))
		
		#Test for a cube
		theta = np.pi/2. - np.arctan(1./np.sqrt(2))
		phi = np.pi/4
		self.assertTrue(np.allclose(geo.spherical_to_cartesian(r,theta,phi),[1./np.sqrt(3),1./np.sqrt(3),1./np.sqrt(3)]))
		
		r = np.array([1.,1.])
		theta = np.array([np.pi/2,np.pi/2. - np.arctan(1./np.sqrt(2))])
		phi = np.array([0.,np.pi/4.])
		self.assertTrue(np.allclose(geo.spherical_to_cartesian(r,theta,phi)[0],np.array([1.,1/np.sqrt(3)])))
		
		
	def test_plan_cartesian_equation(self):
		# plan parallel to (yOz)
		d = 10.
		theta_0 = np.pi/2
		phi_0 = 0
		result = np.array([-1,0,0,10])
		self.assertTrue(np.allclose(geo.plan_cartesian_equation(theta_0,phi_0,d),result))
		
		d = 5.
		theta_0 = np.pi/2
		phi_0 = np.pi/2
		result = np.array([0,-1,0,5])
		self.assertTrue(np.allclose(geo.plan_cartesian_equation(theta_0,phi_0,d),result))
		
		"""d = 3.
		theta_0 = np.pi/6
		phi_0 = np.pi/4
		a,b,c,d = geo.plan_cartesian_equation(theta_0,phi_0,d)
		#print(a,b,c,d)
		geo.plot_plan(a,b,c,d)"""
		
	
	def test_planar_to_gradient(self):
		
		X = np.random.rand(4,2)
		geo.planar_to_gradient(X[:,0], X[:,1])
