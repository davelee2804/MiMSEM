#!/usr/bin/env python

import sys
import numpy as np
#from scipy.spatial import SphericalVoronoi
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
import h5py
from Geom2 import init_geom

pn = 3
ne = 12
print 'writing image'
xg, yg, zg = init_geom(pn,ne,False)
print '...done'

filename = sys.argv[1]
fieldname = sys.argv[2]
print 'file to read ' + filename
f = h5py.File(filename,'r')
print f
print f.name
print f.keys()
print f.values()[0]
w = f[fieldname][()]
print w.shape

xlen = xg.shape[0]
X = np.zeros((xlen,3),dtype=np.float64)
theta = np.zeros((xlen),dtype=np.float64)
phi = np.zeros((xlen),dtype=np.float64)
orig = np.zeros(3,dtype=np.float64)
for ii in np.arange(xlen):
	theta[ii] = np.arctan2(yg[ii],xg[ii])
        phi[ii] = np.arcsin(zg[ii])
	X[ii][0] = xg[ii]
	X[ii][1] = yg[ii]
	X[ii][2] = zg[ii]

triang = mtri.Triangulation(theta,phi)
fig = plt.figure()
plt.tricontourf(triang, w, 100)
plt.triplot(triang)
plt.colorbar()
plt.show()
