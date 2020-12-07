#!/usr/bin/env python

import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
import h5py
from Geom2 import *

pn = 3
ne = 6
print 'writing image'
xg, yg, zg = init_geom(pn,ne,False,False)
print '...done'

filename = sys.argv[1]
fieldname = sys.argv[2]
tmin = float(sys.argv[3])
tmax = float(sys.argv[4])
print 'file to read ' + filename
#f = h5py.File(filename,'r')
#print f
#print f.name
#print f.keys()
#print f.values()[0]
##w = f[fieldname][()]
#w = f[fieldname][:]
#print w.shape
#print xg.shape

#create the mesh
xlen = xg.shape[0]
theta = np.zeros((xlen),dtype=np.float64)
phi = np.zeros((xlen),dtype=np.float64)
for ii in np.arange(xlen):
	theta[ii] = np.arctan2(yg[ii],xg[ii])
	phi[ii] = np.arcsin(zg[ii])

triang = mtri.Triangulation(theta,phi)

#delete text lines
for ii in np.arange(51):
	step = "%.4d"%ii
	filename2 = filename + '_' + step
	print filename2

	f = open(filename2 + '.dat','r')
	lines = f.readlines()
	f.close()
	f = open(filename2 + '.dat','w')
	for line in lines:
		if line[0] != ' ' and line[0] != 'V' and line[0] != 'P':
			f.write(line)
	f.close()

	w=np.loadtxt(filename2 + '.dat')

	#X = np.zeros((xlen,3),dtype=np.float64)
	W = np.zeros(xlen,dtype=np.float64)
	for ii in np.arange(xlen):
		#X[ii][0] = xg[ii]
		#X[ii][1] = yg[ii]
		#X[ii][2] = zg[ii]
		#W[ii] = w[ii][0]
		W[ii] = w[ii]

	fig = plt.figure()
	if np.abs(tmin) < 1.0e-6 and np.abs(tmax) < 1.0e-6:
		plt.tricontourf(triang, W, 100)
	else:
		levs=np.linspace(tmin,tmax,100,endpoint=True)
		plt.tricontourf(triang, W, levs)
	#plt.triplot(triang)
	plt.colorbar()
	plt.savefig(filename2 + '.png')
	#plt.show()
