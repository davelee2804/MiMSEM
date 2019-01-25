#!/usr/bin/env python

import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
#import h5py
from Geom2 import *

pn = 3
ne = int(sys.argv[2])
print 'writing image'
xg, yg, zg = init_geom(pn,ne,False,False)
print '...done'

filename = sys.argv[1]
fieldname = sys.argv[2]
print 'file to read ' + filename

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
plt.tricontourf(triang, W, 100)
plt.colorbar(orientation='horizontal')
plt.xlim([-np.pi,+np.pi])
plt.ylim([-0.5*np.pi,+0.5*np.pi])
plt.savefig(filename + '.png')
plt.show()
