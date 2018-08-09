#!/usr/bin/env python

import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
import h5py
from Geom2 import *

pn = 3
ne = 4
nk = 10
print 'generate geometry...'
xg, yg, zg = init_geom(pn,ne,False,False)

xlen = xg.shape[0]
X = np.zeros((xlen,3),dtype=np.float64)
theta = np.zeros((xlen),dtype=np.float64)
phi = np.zeros((xlen),dtype=np.float64)
for ii in np.arange(xlen):
	theta[ii] = np.arctan2(yg[ii],xg[ii])
	phi[ii] = np.arcsin(zg[ii])
	X[ii][0] = xg[ii]
	X[ii][1] = yg[ii]
	X[ii][2] = zg[ii]

triang = mtri.Triangulation(theta,phi)


print '\t...done'

step = int(sys.argv[1])
fieldnames = ['velocity_h_x_','velocity_h_y_','vorticity_','exner_','rhoTheta_','density_']
for fieldname in fieldnames:
	for kk in np.arange(nk):
		filename = 'output/' + fieldname + '%.3d'%kk + '_%.4d'%step + '.dat'
		print filename

		#delete text lines
		f = open(filename,'r')
		lines = f.readlines()
		f.close()
		f = open(filename,'w')
		for line in lines:
			if line[0] != ' ' and line[0] != 'V' and line[0] != 'P':
				f.write(line)
		f.close()

		w=np.loadtxt(filename)

		picname = 'output/' + fieldname + '%.3d'%kk + '_%.4d'%step + '.png'

		fig = plt.figure()
		plt.tricontourf(triang, w, 100)
		plt.colorbar(orientation='horizontal')
		plt.xlim([-np.pi,+np.pi])
		plt.ylim([-0.5*np.pi,+0.5*np.pi])
		plt.savefig(picname)

fieldnames = ['velocity_z_']
for fieldname in fieldnames:
	for kk in np.arange(nk-1):
		filename = 'output/' + fieldname + '%.3d'%kk + '_%.4d'%step + '.dat'
		print filename

		#delete text lines
		f = open(filename,'r')
		lines = f.readlines()
		f.close()
		f = open(filename,'w')
		for line in lines:
			if line[0] != ' ' and line[0] != 'V' and line[0] != 'P':
				f.write(line)
		f.close()

		w=np.loadtxt(filename)

		picname = 'output/' + fieldname + '%.3d'%kk + '_%.4d'%step + '.png'

		fig = plt.figure()
		plt.tricontourf(triang, w, 100)
		plt.colorbar(orientation='horizontal')
		plt.xlim([-np.pi,+np.pi])
		plt.ylim([-0.5*np.pi,+0.5*np.pi])
		plt.savefig(picname)

