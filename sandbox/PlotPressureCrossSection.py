#!/usr/bin/env python3

import sys
import numpy as np
import scipy.interpolate
import scipy.spatial
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
from Geom2 import *

plt.rcParams['image.cmap'] = 'jet'

pn = 3
ne = 24
nk = 30
print('generate geometry...')
xg, yg, zg = init_geom(pn,ne,False,False)

xlen = xg.shape[0]
theta = np.zeros((xlen),dtype=np.float64)
phi = np.zeros((xlen),dtype=np.float64)
for ii in np.arange(xlen):
	theta[ii] = np.arctan2(yg[ii],xg[ii])
	phi[ii] = np.arcsin(zg[ii])

triang = mtri.Triangulation(theta,phi)
#angs = zip(theta,phi)
angs = np.zeros((len(theta),2))
angs[:,0] = theta
angs[:,1] = phi
delaunay = scipy.spatial.Delaunay(angs)

print( '\t...done')

nx = 64
ny = 128
x = np.linspace(-0.75*np.pi, +0.75*np.pi, nx)
y = np.linspace(-0.45*np.pi, +0.45*np.pi, ny)
x, y = np.meshgrid(x, y)

x2 = np.linspace(-0.98*np.pi, +0.98*np.pi, 256)
yn = (+0.5*np.pi*5.0/9.0)*np.ones((256))
ys = (-0.5*np.pi*5.0/9.0)*np.ones((256))

step = int(sys.argv[1])
ex = np.zeros((nk,256));
for kk in np.arange(nk):
	filename = './output/exner_' + '%.3d'%kk + '_%.4d'%step + '.dat'

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
	interp = scipy.interpolate.LinearNDInterpolator(delaunay, w, fill_value=0)

	wn = 1000.0*np.power(interp(x2, yn)/1004.5,1004.5/287.0)
	ws = 1000.0*np.power(interp(x2, ys)/1004.5,1004.5/287.0)
	ex[kk,:] = wn - np.sum(ws)/len(x2)

ztop = 30000.0
mu = 15.0
Z = np.zeros(nk+1)
Zh = np.zeros(nk)
for ii in np.arange(nk+1):
	frac = (1.0*ii)/nk
	Z[ii] = ztop*(np.sqrt(mu*frac*frac + 1.0) - 1.0)/(np.sqrt(mu + 1.0) - 1.0)

for ii in np.arange(nk):
	Zh[ii] = 0.5*(Z[ii]+Z[ii+1])

fig = plt.figure()
tmp = ex[:,:]
tmp.flatten()
#t_min = np.min(tmp)
#t_max = np.max(tmp)
t_min = -32.0
t_max = +18.0
plt.contourf(x2,Zh,ex,100,levels=np.linspace(t_min,t_max,26,endpoint=True))
levs = np.linspace(t_min, t_max, 6, endpoint=True)
plt.colorbar(orientation='horizontal', ticks=levs)
levs = np.linspace(t_min, t_max, 26, endpoint=True)
plt.contour(x2,Zh,ex,levs,colors='k')
plt.ylim([0.0,15000.0])
picname = './pressure_' + '%.4d'%step + '_merid_avg.png'
plt.savefig(picname)

