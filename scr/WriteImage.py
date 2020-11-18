#!/usr/bin/env python3

import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
#import h5py
from Geom2 import *

pn = 2
ne = 12
print('writing image')
xg, yg, zg = init_geom(pn,ne,False,False)
print('...done')

filename = sys.argv[1]
tmin = float(sys.argv[2])
tmax = float(sys.argv[3])
print('file to read ' + filename)
#f = h5py.File(filename,'r')
#print f
#print f.name
#print f.keys()
#print f.values()[0]
#print w.shape
#print xg.shape

#delete text lines
f = open(filename + '.dat','r')
lines = f.readlines()
f.close()
f = open(filename + '.dat','w')
for line in lines:
	if line[0] != ' ' and line[0] != 'V' and line[0] != 'P':
		f.write(line)
f.close()

w=np.loadtxt(filename + '.dat')

xlen = xg.shape[0]
X = np.zeros((xlen,3),dtype=np.float64)
theta = np.zeros((xlen),dtype=np.float64)
phi = np.zeros((xlen),dtype=np.float64)
W = np.zeros(xlen,dtype=np.float64)
for ii in np.arange(xlen):
	theta[ii] = np.arctan2(yg[ii],xg[ii])
	phi[ii] = np.arcsin(zg[ii])
	X[ii][0] = xg[ii]
	X[ii][1] = yg[ii]
	X[ii][2] = zg[ii]
	#W[ii] = w[ii][0]
	W[ii] = w[ii]

triang = mtri.Triangulation(theta,phi)
fig = plt.figure()
if np.abs(tmin) < 1.0e-6 and np.abs(tmax) < 1.0e-6:
	plt.tricontourf(triang, W, 100)
else:
	levs=np.linspace(tmin,tmax,100,endpoint=True)
	plt.tricontourf(triang, W, levs)
#plt.triplot(triang)
plt.colorbar(orientation='horizontal')
#tc=plt.tricontour(triang,W,1000.0*np.array([9.0,9.2,9.4,9.6,9.8,10.0]),colors='k')
plt.xlim([-np.pi,+np.pi])
plt.ylim([-0.5*np.pi,+0.5*np.pi])
plt.savefig(filename + '.png')
plt.show()
