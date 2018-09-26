#!/usr/bin/env python

import sys
import numpy as np
import scipy.interpolate
import scipy.spatial
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
from Geom2 import *

pn = 3
ne = 24
nk = 20
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

x2 = np.linspace(-0.98*np.pi, +0.98*np.pi, 256)
yn = (+0.5*np.pi*5.0/9.0)*np.ones((256))
ys = (-0.5*np.pi*5.0/9.0)*np.ones((256))

step = int(sys.argv[1])
fieldnames = ['velocity_h_x_','velocity_h_y_','vorticity_','exner_','rhoTheta_','density_','kinEn_']
yz = np.zeros((len(fieldnames),nk,ny))
ex = np.zeros((nk,256));
field_i = 0
for fieldname in fieldnames:
	for kk in np.arange(nk):
		filename = './' + fieldname + '%.3d'%kk + '_%.4d'%step + '.dat'
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

		picname = './' + fieldname + '%.3d'%kk + '_%.4d'%step + '.png'

		tmp = w[:]
		tmp.flatten()
		t_min = np.min(tmp)
		t_max = np.max(tmp)
		levs = np.linspace(t_min, t_max, 13, endpoint=True)

		fig = plt.figure()
		plt.tricontourf(triang, w, 100)
		if fieldname != 'vorticity_':
			plt.colorbar(orientation='horizontal')
			plt.tricontour(triang, w, levs, colors='k')
		else:
			levs = np.linspace(t_min, t_max, 5, endpoint=True)
			plt.colorbar(orientation='horizontal', format='%.1e', ticks=levs)
		plt.xlim([-np.pi,+np.pi])
		plt.ylim([-0.5*np.pi,+0.5*np.pi])
		plt.savefig(picname)

		interp = scipy.interpolate.LinearNDInterpolator(delaunay, w, fill_value=0)
		wc = interp(x, y)
		for yy in np.arange(ny):
			yz[field_i,kk,yy] = np.sum(wc[yy,:])/nx

		if fieldname == 'exner_':
			wn = 100000.0*np.power(interp(x2, yn)/1004.5,-287.0/1004.5)
			ws = 100000.0*np.power(interp(x2, ys)/1004.5,-287.0/1004.5)
			ex[kk,:] = wn - np.sum(ws)/len(x2)

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

fig = plt.figure()
tmp = ex[:,:]
tmp.flatten()
t_min = np.min(tmp)
t_max = np.max(tmp)
levs = np.linspace(t_min, t_max, 13, endpoint=True)
plt.contourf(x2,Zh,ex,100)
plt.colorbar(orientation='horizontal')
plt.contour(x2,Zh,ex,levs,colors='k')
picname = './pressure_' + '%.4d'%step + '_merid_avg.png'
plt.savefig(picname)

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
	picname = './' + fieldnames[field_i] + '%.4d'%step + '_zonal_avg.png'
	plt.savefig(picname)

fieldnames = ['velocity_z_']
velz_yz = np.zeros((nk-1,ny))
for fieldname in fieldnames:
	for kk in np.arange(nk-1):
		filename = './' + fieldname + '%.3d'%kk + '_%.4d'%step + '.dat'
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

		picname = './' + fieldname + '%.3d'%kk + '_%.4d'%step + '.png'

		tmp = w[:]
		tmp.flatten()
		t_min = np.min(tmp)
		t_max = np.max(tmp)
		if(np.abs(t_min) > t_max):
			t_max = np.abs(t_min)

		#levs = np.linspace(t_min, t_max, 13, endpoint=True)
		#levs2 = np.zeros(12)
		#ii = 0
		#for jj in np.arange(13):
		#	if np.abs(levs[jj]) > 1.0e-5:
		#		levs2[ii] = levs[jj]
		#		ii = ii + 1

		fig = plt.figure()
		plt.tricontourf(triang, w, 100)
		plt.colorbar(orientation='horizontal')
		#plt.tricontour(triang, w, levs2, colors='k')
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
picname = './velocity_z_' + '%.4d'%step + '_zonal_avg.png'
plt.savefig(picname)

fieldnames = ['theta_']
theta_yz = np.zeros((nk+1,ny))
for fieldname in fieldnames:
	for kk in np.arange(nk+1):
		filename = './' + fieldname + '%.3d'%kk + '_%.4d'%step + '.dat'
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

		tmp = w[:]
		tmp.flatten()
		t_min = np.min(tmp)
		t_max = np.max(tmp)
		levs = np.linspace(t_min, t_max, 13, endpoint=True)

		picname = './' + fieldname + '%.3d'%kk + '_%.4d'%step + '.png'

		fig = plt.figure()
		plt.tricontourf(triang, w, 100)
		plt.colorbar(orientation='horizontal')
		plt.tricontour(triang, w, levs, colors='k')
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
picname = './theta_' + '%.4d'%step + '_zonal_avg.png'
plt.savefig(picname)

