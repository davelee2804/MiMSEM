#!/usr/bin/env python

import sys
import numpy as np
import scipy.interpolate
import scipy.spatial
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
#import h5py
from Geom2 import *

pn = 3
ne = 16
nk = 30
path = 'output/'
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
angs = zip(theta,phi)
delaunay = scipy.spatial.Delaunay(angs)

print '\t...done'

nx = 64
ny = 128
x = np.linspace(-0.75*np.pi, +0.75*np.pi, nx)
y = np.linspace(-0.45*np.pi, +0.45*np.pi, ny)
x, y = np.meshgrid(x, y)


step = int(sys.argv[1])
fieldnames = ['velocity_h_x_','velocity_h_y_','vorticity_','exner_','rhoTheta_','density_','ke_']
yz = np.zeros((len(fieldnames),nk,ny))
field_i = 0
for fieldname in fieldnames:
	for kk in np.arange(nk):
		filename = path + fieldname + '%.3d'%kk + '_%.4d'%step + '.dat'
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

		picname = path + fieldname + '%.3d'%kk + '_%.4d'%step + '.png'

		fig = plt.figure()
		plt.tricontourf(triang, w, 100)
		plt.colorbar(orientation='horizontal')
		plt.xlim([-np.pi,+np.pi])
		plt.ylim([-0.5*np.pi,+0.5*np.pi])
		plt.savefig(picname)

		interp = scipy.interpolate.LinearNDInterpolator(delaunay, w, fill_value=0)
		wc = interp(x, y)
		for yy in np.arange(ny):
			yz[field_i,kk,yy] = np.sum(wc[yy,:])/nx

	field_i = field_i + 1

ztop = 30000.0
mu = 15.0
Z = np.zeros(nk+1)
Zh = np.zeros(nk)
for ii in np.arange(nk+1):
	frac = (1.0*ii)/nk
	Z[ii] = ztop*(np.sqrt(mu*frac*frac + 1.0) - 1.0)/(np.sqrt(mu + 1.0) - 1.0)

for ii in np.arange(nk):
	Zh[ii] = 0.5*(Z[ii]+Z[ii+1])

for field_i in np.arange(len(fieldnames)):
	tmp = yz[field_i,:,:]
	tmp.flatten()
	t_min = np.min(tmp)
	t_max = np.max(tmp)
	levs = np.linspace(t_min, t_max, 21, endpoint=True)

	fig = plt.figure()
	plt.contourf(np.linspace(-0.4*np.pi,+0.4*np.pi,ny),Zh,yz[field_i,:,:],100)
	plt.colorbar(orientation='horizontal')
	plt.contour(np.linspace(-0.4*np.pi,+0.4*np.pi,ny),Zh,yz[field_i,:,:],levs,colors='k')
	picname = path + fieldnames[field_i] + '%.4d'%step + '_zonal_avg.png'
	plt.savefig(picname)

fieldnames = ['velocity_z_']
velz_yz = np.zeros((nk-1,ny))
for fieldname in fieldnames:
	for kk in np.arange(nk-1):
		filename = path + fieldname + '%.3d'%kk + '_%.4d'%step + '.dat'
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

		picname = path + fieldname + '%.3d'%kk + '_%.4d'%step + '.png'

		fig = plt.figure()
		plt.tricontourf(triang, w, 100)
		plt.colorbar(orientation='horizontal')
		plt.xlim([-np.pi,+np.pi])
		plt.ylim([-0.5*np.pi,+0.5*np.pi])
		plt.savefig(picname)

		interp = scipy.interpolate.LinearNDInterpolator(delaunay, w, fill_value=0)
		wc = interp(x, y)
		for yy in np.arange(ny):
			velz_yz[kk,yy] = np.sum(wc[yy,:])/nx

tmp = velz_yz[:,:]
tmp.flatten()
t_min = np.min(tmp)
t_max = np.max(tmp)
levs = np.linspace(t_min, t_max, 21, endpoint=True)

fig = plt.figure()
plt.contourf(np.linspace(-0.4*np.pi,+0.4*np.pi,ny),Z[1:-1],velz_yz[:,:],100)
plt.colorbar(orientation='horizontal')
plt.contour(np.linspace(-0.4*np.pi,+0.4*np.pi,ny),Z[1:-1],velz_yz[:,:],levs,colors='k')
picname = path + 'velocity_z_' + '%.4d'%step + '_zonal_avg.png'
plt.savefig(picname)

fieldnames = ['theta_']
theta_yz = np.zeros((nk+1,ny))
for fieldname in fieldnames:
	for kk in np.arange(nk+1):
		filename = path + fieldname + '%.3d'%kk + '_%.4d'%step + '.dat'
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

		picname = path + fieldname + '%.3d'%kk + '_%.4d'%step + '.png'

		fig = plt.figure()
		plt.tricontourf(triang, w, 100)
		plt.colorbar(orientation='horizontal')
		plt.xlim([-np.pi,+np.pi])
		plt.ylim([-0.5*np.pi,+0.5*np.pi])
		plt.savefig(picname)

		interp = scipy.interpolate.LinearNDInterpolator(delaunay, w, fill_value=0)
		wc = interp(x, y)
		for yy in np.arange(ny):
			theta_yz[kk,yy] = np.sum(wc[yy,:])/nx

tmp = theta_yz[:,:]
tmp.flatten()
t_min = np.min(tmp)
t_max = np.max(tmp)
levs = np.linspace(t_min, t_max, 21, endpoint=True)

fig = plt.figure()
plt.contourf(np.linspace(-0.4*np.pi,+0.4*np.pi,ny),Z,theta_yz[:,:],100)
plt.colorbar(orientation='horizontal')
plt.contour(np.linspace(-0.4*np.pi,+0.4*np.pi,ny),Z,theta_yz[:,:],levs,colors='k')
picname = path + 'theta_' + '%.4d'%step + '_zonal_avg.png'
plt.savefig(picname)

