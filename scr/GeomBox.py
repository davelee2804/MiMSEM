#!/usr/bin/env python

import sys
import numpy as np
#from matplotlib import pyplot as plt

from ProcBox import *

def init_geom(_pn, _ne, make_image, lx):
	pn = int(_pn)
	ne = int(_ne)

	a = np.sqrt(7.0)
	b = np.sqrt((7.0 - 2.0*a)/21.0)
	c = np.sqrt((7.0 + 2.0*a)/21.0)

	q1 = np.array([-1.0,+1.0])
	q2 = np.array([-1.0,0.0,+1.0])
	q3 = np.array([-1.0,-np.sqrt(0.2),+np.sqrt(0.2),+1.0])
	q4 = np.array([-1,-np.sqrt(3.0/7.0),0.0,+np.sqrt(3.0/7.0),+1])
	q5 = np.array([-1.0,-c,-b,+b,+c,+1.0])

	if pn == 1:
		qn = q1
	elif pn == 2:
		qn = q2
	elif pn == 3:
		qn = q3
	elif pn == 4:
		qn = q4
	elif pn == 5:
		qn = q5
	else:
		print("invalid polynomial order: " + str(pn))

	nx = pn*ne
	dx = lx/ne
	
	if make_image:
		fig = plt.figure()
		ax = Axes3D(fig)

	x0 = np.zeros(nx*nx)
	y0 = np.zeros(nx*nx)
	z0 = np.zeros(nx*nx)

	ii = 0
	for iy in np.arange(ne*pn):
		for ix in np.arange(ne*pn):
			x0[ii] = (ix//pn)*dx + 0.5*dx*(1.0 + qn[ix%pn])
			y0[ii] = (iy//pn)*dx + 0.5*dx*(1.0 + qn[iy%pn])
			z0[ii] = 0.0
			ii = ii + 1

	if make_image:
		ax.scatter(x0[:nx*nx], y0[:nx*nx], z0[:nx*nx], c='g')
		#plot the first hanging node
		ax.scatter(x0[nx*nx], y0[nx*nx], z0[nx*nx], c='g', marker='s')

		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()

	if make_image:
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(x0,y0,z0,c='g')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()

	return x0, y0, z0
