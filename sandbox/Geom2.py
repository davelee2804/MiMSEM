#!/usr/bin/env python

import sys
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from Proc2 import *

def init_geom(pn, ne, make_image, expand_sphere):
	a = np.sqrt(7.0)
	b = np.sqrt((7.0 - 2.0*a)/21.0)
	c = np.sqrt((7.0 + 2.0*a)/21.0)

	q2 = np.array([-1.0,0.0,+1.0])
	q3 = np.array([-1.0,-np.sqrt(0.2),+np.sqrt(0.2),+1.0])
	q4 = np.array([-1,-np.sqrt(3.0/7.0),0.0,+np.sqrt(3.0/7.0),+1])
	q5 = np.array([-1.0,-c,-b,+b,+c,+1.0])

	nx = pn*ne
	X = np.zeros(pn*ne+1,dtype=np.float64)
	dx = 0.5*np.pi/ne
	for el in np.arange(ne):
		if pn == 2:
			X[el*pn:(el+1)*pn] = dx*0.5*(q2[:pn]+1.0) + el*dx - 0.25*np.pi
		if pn == 3:
			X[el*pn:(el+1)*pn] = dx*0.5*(q3[:pn]+1.0) + el*dx - 0.25*np.pi
		if pn == 4:
			X[el*pn:(el+1)*pn] = dx*0.5*(q4[:pn]+1.0) + el*dx - 0.25*np.pi
		elif pn == 5:
			X[el*pn:(el+1)*pn] = dx*0.5*(q5[:pn]+1.0) + el*dx - 0.25*np.pi
	
		X[pn*ne] = +0.25*np.pi
	
	if make_image:
		fig = plt.figure()
		ax = Axes3D(fig)

	x0 = np.zeros(nx*nx+1)
	y0 = np.zeros(nx*nx+1)
	z0 = np.zeros(nx*nx+1)

	ii = 0
	for iy in np.arange(ne*pn):
		for ix in np.arange(ne*pn):
			tx = np.tan(X[ix])
			ty = np.tan(X[iy])

			theta = X[ix]
			phi = np.arcsin(ty/np.sqrt(1.0 + tx*tx + ty*ty))

			x0[ii] = +np.cos(phi)*np.cos(theta)
			y0[ii] = +np.cos(phi)*np.sin(theta)
			z0[ii] = +np.sin(phi)
			ii = ii + 1

	#add in the first hanging node
	tx = np.tan(X[0])
	ty = np.tan(X[0])
	theta = X[nx]
	phi = np.arcsin(ty/np.sqrt(1.0 + tx*tx + ty*ty))
	x0[nx*nx] = np.cos(phi)*np.cos(theta)
	y0[nx*nx] = np.cos(phi)*np.sin(theta)
	z0[nx*nx] = np.sin(phi)

	if make_image:
		ax.scatter(x0[:nx*nx], y0[:nx*nx], z0[:nx*nx], c='g')
		#plot the first hanging node
		ax.scatter(x0[nx*nx], y0[nx*nx], z0[nx*nx], c='g', marker='s')

	# rotate north +pi/2
	A1 = np.zeros((3,3),dtype=np.int8)
	A1[0,2] = -1
	A1[1,1] = +1
	A1[2,0] = +1

	x1 = np.zeros(nx*nx+1)
	y1 = np.zeros(nx*nx+1)
	z1 = np.zeros(nx*nx+1)
	ii = 0
	for iy in np.arange(ne*pn):
		for ix in np.arange(ne*pn):
			x1[ii] = A1[0,2]*z0[ii]
			y1[ii] = A1[1,1]*y0[ii]
			z1[ii] = A1[2,0]*x0[ii]
			ii = ii + 1

	# add the second hanging node
	tx = np.tan(X[nx])
	ty = np.tan(X[nx])
	theta = X[0]
	phi = np.arcsin(ty/np.sqrt(1.0 + tx*tx + ty*ty))
	x1[nx*nx] = A1[0,2]*np.sin(phi)
	y1[nx*nx] = A1[1,1]*np.cos(phi)*np.sin(theta)
	z1[nx*nx] = A1[2,0]*np.cos(phi)*np.cos(theta)

	if make_image:
		ax.scatter(x1[:nx*nx], y1[:nx*nx], z1[:nx*nx], c='r')
                #jj=[str(a) for a in np.linspace(0.0,1.0,nx*nx)]
		#ax.scatter(x1[:nx*nx], y1[:nx*nx], z1[:nx*nx], c=jj)
		#plot the second hanging node
		ax.scatter(x1[nx*nx], y1[nx*nx], z1[nx*nx], c='r', marker='s')

	# rotate east +pi/2
	A2 = np.zeros((3,3),dtype=np.int8)
	A2[0,0] = +1
	A2[1,2] = +1
	A2[2,1] = -1

	x2 = np.zeros(nx*nx)
	y2 = np.zeros(nx*nx)
	z2 = np.zeros(nx*nx)
	ii = 0
	for iy in np.arange(ne*pn):
		for ix in np.arange(ne*pn):
			x2[ii] = A2[0,0]*x1[ii]
			y2[ii] = A2[1,2]*z1[ii]
			z2[ii] = A2[2,1]*y1[ii]
			ii = ii + 1

	if make_image:
		ax.scatter(x2, y2, z2, c='b')

	# rotate north +pi/2
	A3 = np.zeros((3,3),dtype=np.int8)
	A3[0,1] = -1
	A3[1,0] = +1
	A3[2,2] = +1

	x3 = np.zeros(nx*nx)
	y3 = np.zeros(nx*nx)
	z3 = np.zeros(nx*nx)
	ii = 0
	for iy in np.arange(ne*pn):
		for ix in np.arange(ne*pn):
			x3[ii] = A3[0,1]*y2[ii]
			y3[ii] = A3[1,0]*x2[ii]
			z3[ii] = A3[2,2]*z2[ii]
			ii = ii + 1

	if make_image:
		ax.scatter(x3, y3, z3, c='c')

	# rotate east +pi/2
	A4 = np.zeros((3,3),dtype=np.int8)
	A4[0,2] = -1
	A4[1,1] = +1
	A4[2,0] = +1

	x4 = np.zeros(nx*nx)
	y4 = np.zeros(nx*nx)
	z4 = np.zeros(nx*nx)
	ii = 0
	for iy in np.arange(ne*pn):
		for ix in np.arange(ne*pn):
			x4[ii] = A4[0,2]*z3[ii]
			y4[ii] = A4[1,1]*y3[ii]
			z4[ii] = A4[2,0]*x3[ii]
			ii = ii + 1

	if make_image:
		ax.scatter(x4, y4, z4, c='m')

	# rotate north +pi/2
	A5 = np.zeros((3,3),dtype=np.int8)
	A5[0,0] = +1
	A5[1,2] = +1
	A5[2,1] = -1

	x5 = np.zeros(nx*nx)
	y5 = np.zeros(nx*nx)
	z5 = np.zeros(nx*nx)
	ii = 0
	for iy in np.arange(ne*pn):
		for ix in np.arange(ne*pn):
			x5[ii] = A5[0,0]*x4[ii]
			y5[ii] = A5[1,2]*z4[ii]
			z5[ii] = A5[2,1]*y4[ii]
			ii = ii + 1

	if make_image:
		ax.scatter(x5, y5, z5, c='y')

		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()

	xg = np.zeros(6*nx*nx+2,dtype=np.float64)
	yg = np.zeros(6*nx*nx+2,dtype=np.float64)
	zg = np.zeros(6*nx*nx+2,dtype=np.float64)

	xg[0*nx*nx:1*nx*nx] = x0[:nx*nx]
	yg[0*nx*nx:1*nx*nx] = y0[:nx*nx]
	zg[0*nx*nx:1*nx*nx] = z0[:nx*nx]
	xg[1*nx*nx:2*nx*nx] = x1[:nx*nx]
	yg[1*nx*nx:2*nx*nx] = y1[:nx*nx]
	zg[1*nx*nx:2*nx*nx] = z1[:nx*nx]
	xg[2*nx*nx:3*nx*nx] = x2[:nx*nx]
	yg[2*nx*nx:3*nx*nx] = y2[:nx*nx]
	zg[2*nx*nx:3*nx*nx] = z2[:nx*nx]
	xg[3*nx*nx:4*nx*nx] = x3[:nx*nx]
	yg[3*nx*nx:4*nx*nx] = y3[:nx*nx]
	zg[3*nx*nx:4*nx*nx] = z3[:nx*nx]
	xg[4*nx*nx:5*nx*nx] = x4[:nx*nx]
	yg[4*nx*nx:5*nx*nx] = y4[:nx*nx]
	zg[4*nx*nx:5*nx*nx] = z4[:nx*nx]
	xg[5*nx*nx:6*nx*nx] = x5[:nx*nx]
	yg[5*nx*nx:6*nx*nx] = y5[:nx*nx]
	zg[5*nx*nx:6*nx*nx] = z5[:nx*nx]
	xg[6*nx*nx+0] = x0[nx*nx]
	yg[6*nx*nx+0] = y0[nx*nx]
	zg[6*nx*nx+0] = z0[nx*nx]
	xg[6*nx*nx+1] = x1[nx*nx]
	yg[6*nx*nx+1] = y1[nx*nx]
	zg[6*nx*nx+1] = z1[nx*nx]

	#Rx = np.zeros((3,3),dtype=np.float64)
	#theta = 0.25*np.pi
	#Rx[0][0] = 1.0
	#Rx[1][1] = +np.cos(theta)
	#Rx[1][2] = -np.sin(theta)
	#Rx[2][1] = +np.sin(theta)
	#Rx[2][2] = +np.cos(theta)
	#Ry = np.zeros((3,3),dtype=np.float64)
	#Ry[0][0] = -np.sin(theta)
	#Ry[0][2] = +np.cos(theta)
	#Ry[1][1] = 1.0
	#Ry[2][0] = +np.cos(theta)
	#Ry[2][2] = +np.sin(theta)
	#Rxy = np.matmul(Ry,Rx)

	#for ii in np.arange(6*nx*nx+2):
	#	xk = np.array([xg[ii],yg[ii],zg[ii]],dtype=np.float64)
	#	xj = np.zeros((3),dtype=np.float64)
	#	for jj in np.arange(3):
	#		for kk in np.arange(3):
	#			xj[jj] = xj[jj] + Rxy[jj][kk]*xk[kk]

	#	xg[ii] = xj[0]
	#	yg[ii] = xj[1]
	#	zg[ii] = xj[2]

	if expand_sphere:
		re = 6371220.0
		for ii in np.arange(6*nx*nx+2):
			theta = np.arctan2(yg[ii],xg[ii]);
			phi = np.arcsin(zg[ii])

			xg[ii] = re*np.cos(phi)*np.cos(theta)
			yg[ii] = re*np.cos(phi)*np.sin(theta)
			zg[ii] = re*np.sin(phi)

	if make_image:
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(xg[0*nx*nx:1*nx*nx],yg[0*nx*nx:1*nx*nx],zg[0*nx*nx:1*nx*nx],c='g')
		ax.scatter(xg[1*nx*nx:2*nx*nx],yg[1*nx*nx:2*nx*nx],zg[1*nx*nx:2*nx*nx],c='r')
		ax.scatter(xg[2*nx*nx:3*nx*nx],yg[2*nx*nx:3*nx*nx],zg[2*nx*nx:3*nx*nx],c='b')
		ax.scatter(xg[3*nx*nx:4*nx*nx],yg[3*nx*nx:4*nx*nx],zg[3*nx*nx:4*nx*nx],c='c')
		ax.scatter(xg[4*nx*nx:5*nx*nx],yg[4*nx*nx:5*nx*nx],zg[4*nx*nx:5*nx*nx],c='m')
		ax.scatter(xg[5*nx*nx:6*nx*nx],yg[5*nx*nx:6*nx*nx],zg[5*nx*nx:6*nx*nx],c='y')
		ax.scatter(xg[6*nx*nx+0],yg[6*nx*nx+0],zg[6*nx*nx+0],c='g',marker='s')
		ax.scatter(xg[6*nx*nx+1],yg[6*nx*nx+1],zg[6*nx*nx+1],c='r',marker='s')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		plt.show()

	return xg, yg, zg
