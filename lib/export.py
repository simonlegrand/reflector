"""
This file contains function to write triangulation
into a .off file.
"""
from __future__ import print_function

import sys
import time

sys.path.append('../PyMongeAmpere-build/')
sys.path.append('../PyMongeAmpere-build/lib')
sys.path.append('./lib')

import numpy as np

def write_header(outfile,numVertices,numFaces,numEdges=None):
	"""
	Write header of the file.
	Line 1 : OFF
	Line 2 : numVertices	numFaces	numEdges(optionnal)
	"""	
	outfile.write('OFF'+'\n')
	if numEdges is None:
		outfile.write(str(numVertices)+'\t'+str(numFaces)+"\t"+str(0)+"\n")
		
	else:
		outfile.write(str(numVertices)+'\t'+str(numFaces)+"\t"+str(numEdges)+"\n")


def write_improved_header(outfile,numVertices,numFaces,numEdges=None):
	"""
	Write header of the file.
	Line 1 : OFF
	Line 2 : numVertices	numFaces	numEdges(optionnal)
	"""	
	outfile.write('IOFF'+'\n')
	if numEdges is None:
		outfile.write(str(numVertices)+'\t'+str(numFaces)+"\t"+str(0)+"\n")
		
	else:
		outfile.write(str(numVertices)+'\t'+str(numFaces)+"\t"+str(numEdges)+"\n")


def write_points(outfile, points):
	"""
	Write points in the file.
	One line per point:
	x	y	z
	"""
	for i in range (0, len(points)):
		data = str(points[i,0]) + "\t" + str(points[i,1]) + "\t" + str(points[i,2]) + "\n"
		outfile.write(data)


def write_points_gradients(outfile, points, gradients):
	"""
	Write points and gradients in the file.
	One line per point:
	x	y	z
	"""
	for i in range (0, len(points)):
		data = str(points[i,0]) + "\t" + str(points[i,1]) + "\t" + str(points[i,2]) + "\t" + str(gradients[i,0]) + "\t" + str(gradients[i,1]) + "\n"
		outfile.write(data)
		
			
def write_polygon(outfile, polygon):
	for i in range (0, np.shape(polygon)[0]):
		data = str(np.shape(polygon)[1])
		for j in range (0, np.shape(polygon)[1]):
			data = data + "\t" + str(polygon[i,j])
		data = data + "\n"
		outfile.write(data)
		
		
def export_off(filename, points, polygon):
	try:
		outfile = open(filename,'w')
		write_header(outfile, len(points), len(polygon))
		write_points(outfile,points)
		write_polygon(outfile,polygon)
	except (NameError, IOError) as e:
		print(e)
		sys.exit(1)
		
		
def export_improved_off(filename, points, gradients, polygon):
	try:
		outfile = open(filename,'w')
		write_improved_header(outfile, len(points), len(polygon))
		write_points_gradients(outfile,points, gradients)
		write_polygon(outfile,polygon)
	except (NameError, IOError) as e:
		print(e)
		sys.exit(1)
