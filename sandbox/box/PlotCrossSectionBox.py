#!/usr/bin/env python3

import sys
import numpy as np
import scipy.interpolate
import scipy.spatial
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
from GeomBox import *

plt.rcParams['image.cmap'] = 'jet'

pn = 3
ne = 24
nk = 150
lx = 1000.0
print('generate geometry...')
xg, yg, zg = init_geom(pn,ne,False,lx)
xmin = min(xg)
xmax = max(xg)
ymin = min(yg)
ymax = max(yg)

triang = mtri.Triangulation(xg,yg)
#angs = zip(xg,yg)
angs = np.zeros((len(xg),2))
angs[:,0] = xg
angs[:,1] = yg
delaunay = scipy.spatial.Delaunay(angs)

print( '\t...done')

nx = 100
x = np.linspace(xmin,xmax,100,endpoint=True)
y = np.linspace(xmin,xmax,100,endpoint=True)
x, y = np.meshgrid(x, y)

x2 = np.linspace(xmin,xmax,100,endpoint=True)
yn = 0.5*lx*np.ones(nx)

step = int(sys.argv[1])

Zh=np.linspace(0.0,1500.0,nk+1,endpoint=True)
theta = np.zeros((nk+1,100));
for kk in np.arange(nk+1):
	filename = './output/theta_' + '%.3d'%kk + '_%.4d'%step + '.dat'

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

	theta[kk,:] = interp(x2, yn)

fig = plt.figure()
tmp = theta[:,:]
tmp.flatten()
t_min = np.min(tmp)
t_max = np.max(tmp)
#plt.contourf(x2,Zh,theta,100,levels=np.linspace(t_min,t_max,26,endpoint=True))
plt.contourf(x2,Zh,theta,100)
levs = np.linspace(t_min, t_max, 6, endpoint=True)
plt.colorbar(orientation='vertical', ticks=levs)
#levs = np.linspace(t_min, t_max, 9, endpoint=True)
levs = np.array([300.1, 300.2, 300.3, 300.4, 300.5])
plt.contour(x2,Zh,theta,levs,colors='k')
plt.ylim([0.0,lx])
plt.xlabel(r'x (m)')
plt.ylabel(r'z (m)')
picname = './output/theta_' + '%.4d'%step + '.pdf'
print('writing image: ' + picname)
plt.savefig(picname)

Zh=np.linspace(0.0,1490.0,nk-1,endpoint=True)
velz = np.zeros((nk-1,100));
for kk in np.arange(nk-1):
	filename = './output/velocity_z_' + '%.3d'%kk + '_%.4d'%step + '.dat'

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

	velz[kk,:] = interp(x2, yn)

fig = plt.figure()
tmp = velz[:,:]
tmp.flatten()
t_min = np.min(tmp)
t_max = np.max(tmp)
#plt.contourf(x2,Zh,velz,100,levels=np.linspace(t_min,t_max,26,endpoint=True))
plt.contourf(x2,Zh,velz,100)
levs = np.linspace(t_min, t_max, 6, endpoint=True)
plt.colorbar(orientation='vertical', ticks=levs)
levs = np.linspace(t_min, t_max, 9, endpoint=True)
plt.contour(x2,Zh,velz,levs,colors='k')
plt.ylim([0.0,lx])
picname = './output/velocity_z_' + '%.4d'%step + '.png'
print('writing image: ' + picname)
plt.savefig(picname)

fieldnames = ['velocity_h_x_','velocity_h_y_','vorticity_','exner_','rhoTheta_','density_']
field = np.zeros((nk,100));
Zh=np.linspace(0.0,1500.0,150) + 5.0
for name in fieldnames:
	for kk in np.arange(nk):
		filename = './output/' + name + '%.3d'%kk + '_%.4d'%step + '.dat'

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

		field[kk,:] = interp(x2, yn)

	fig = plt.figure()
	tmp = field[:,:]
	tmp.flatten()
	t_min = np.min(tmp)
	t_max = np.max(tmp)
	#plt.contourf(x2,Zh,field,100,levels=np.linspace(t_min,t_max,26,endpoint=True))
	plt.contourf(x2,Zh,field,100)
	levs = np.linspace(t_min, t_max, 6, endpoint=True)
	plt.colorbar(orientation='vertical', ticks=levs)
	levs = np.linspace(t_min, t_max, 13, endpoint=True)
	plt.contour(x2,Zh,field,levs,colors='k')
	plt.ylim([0.0,lx])
	picname = './output/' + name + '%.4d'%step + '.png'
	print('writing image: ' + picname)
	plt.savefig(picname)

