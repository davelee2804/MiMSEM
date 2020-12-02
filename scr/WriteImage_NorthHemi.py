#!/usr/bin/env python3

import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
from Geom2 import *

plt.rcParams['image.cmap'] = 'jet'

pn = 3
ne = 24
print('writing image')
xg, yg, zg = init_geom(pn,ne,False,False)
print('...done')

path = sys.argv[1]
filename = sys.argv[2]
do_contours = int(sys.argv[3])
level = int(sys.argv[4])
step = int(sys.argv[5])

#delete text lines
full_name = path + '/' + filename + '_%.3u'%level + '_%.4u'%step + '.dat'
print('reading file: ' + full_name)
f = open(full_name,'r')
lines = f.readlines()
f.close()
f = open(full_name,'w')
for line in lines:
	if line[0] != ' ' and line[0] != 'V' and line[0] != 'P':
		f.write(line)
f.close()

w=np.loadtxt(full_name)

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
		xn[jj] = xg[ii]
		yn[jj] = yg[ii]
		W[jj] = w[ii]
		jj = jj + 1;

triang = mtri.Triangulation(xn[:jj],yn[:jj])
fig = plt.figure()
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.tricontourf(triang, W[:jj], 100)

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

theta_b_file = path + '/theta_' + '%.3u'%level + '_%.4u'%step + '.dat'
theta_t_file = path + '/theta_' + '%.3u'%(level+1) + '_%.4u'%step + '.dat'
print('potential temperature at bottom of level: ' + theta_b_file)
print('potential temperature at top of level:    ' + theta_t_file)

f = open(theta_b_file,'r')
lines = f.readlines()
f.close()
f = open(theta_b_file,'w')
for line in lines:
	if line[0] != ' ' and line[0] != 'V' and line[0] != 'P':
		f.write(line)
f.close()
f = open(theta_t_file,'r')
lines = f.readlines()
f.close()
f = open(theta_t_file,'w')
for line in lines:
	if line[0] != ' ' and line[0] != 'V' and line[0] != 'P':
		f.write(line)
f.close()

theta_b = np.loadtxt(theta_b_file)
theta_t = np.loadtxt(theta_t_file)
theta = 0.5*(theta_b + theta_t)
jj = 0
for ii in np.arange(xlen):
	if zg[ii] > 0.0:
		xn[jj] = xg[ii]
		yn[jj] = yg[ii]
		W[jj] = theta[ii]
		jj = jj + 1;

fig = plt.figure()
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.tricontourf(triang, W[:jj], 100)
plt.colorbar(orientation='vertical')

tmp = W[:jj]
tmp.flatten()
t_min = np.min(tmp)
t_max = np.max(tmp)
levs = np.linspace(t_min, t_max, 13, endpoint=True)
plt.tricontour(triang, W[:jj], levs, colors='k')
plt.savefig('theta_' + '%.4d'%step + '_nh.pdf')
plt.show()
