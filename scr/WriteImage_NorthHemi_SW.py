#!/usr/bin/env python3

import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
from Geom2 import *

plt.rcParams['image.cmap'] = 'nipy_spectral'

pn = 3
ne = 32
print('writing image')
xg, yg, zg = init_geom(pn,ne,False,False)
print('...done')

filename = sys.argv[1]
do_contours = int(sys.argv[2])
print('file to read ' + filename)

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
print(xlen)
X = np.zeros((xlen,3),dtype=np.float64)
xn = np.zeros((xlen),dtype=np.float64)
yn = np.zeros((xlen),dtype=np.float64)
theta = np.zeros((xlen),dtype=np.float64)
phi = np.zeros((xlen),dtype=np.float64)
W = np.zeros(xlen,dtype=np.float64)
jj = 0
for ii in np.arange(xlen):
	if zg[ii] > 0.0:
		xn[jj] = -xg[ii]
		yn[jj] = -yg[ii]
		W[jj] = w[ii]
		jj = jj + 1;

triang = mtri.Triangulation(xn[:jj],yn[:jj])
fig = plt.figure()
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
#plt.tricontourf(triang, W[:jj], 100)
plt.tricontourf(triang, W[:jj], levels=np.linspace(-0.000150,+0.000150,101,endpoint=True))

#plt.triplot(triang)
if filename[11:15] == 'pres':
	plt.colorbar(orientation='vertical',ticks=1000.0*np.array([9.0,9.2,9.4,9.6,9.8,10.0]))
	tc=plt.tricontour(triang,W[:jj],1000.0*np.array([9.0,9.2,9.4,9.6,9.8,10.0]),colors='k')
elif filename[11:15] == 'vort':
	plt.colorbar(orientation='vertical',format='%.1e',ticks=1.0e-5*np.array([-12,-8,-4,0,+4,+8,+12]))
	tc=plt.tricontour(triang,W[:jj],1.0e-5*np.array([-12,-10,-8,-6,-4,-2,0,+2,+4,+6,+8,+10,+12]),colors='k')
else:
	plt.colorbar(orientation='vertical')

if do_contours:
	tmp = W[:jj]
	tmp.flatten()
	t_min = np.min(tmp)
	t_max = np.max(tmp)
	levs = np.linspace(t_min, t_max, 13, endpoint=True)
	plt.tricontour(triang, W[:jj], levs, colors='k')

plt.savefig(filename + '_nh.pdf')
plt.show()

