#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt

from Basis import *
from Topo import *
from BoundaryMat import *
from Mats2D import *
from Assembly import *
from Proj import *
from LieDeriv import *
from AdvEqn import *
from ShallowWaterEqn import *

def velx_func(x,y):
	return +1.0*np.cos(x)*np.sin(y)

def vely_func(x,y):
	return -1.0*np.sin(x)*np.cos(y)

def pres_func(x,y):
	xi = 2.0*np.pi/8.0 + 2.0*np.pi/4.0
	xf = 6.0*np.pi/8.0 + 2.0*np.pi/4.0
	yi = 2.0*np.pi/8.0 + 2.0*np.pi/8.0
	yf = 6.0*np.pi/8.0 + 2.0*np.pi/8.0
	xo = 0.5*(xi + xf)
	yo = 0.5*(yi + yf)
	xi = xo - 1.5*(xo - xi)
	xf = xo + 1.5*(xf - xo)
	yi = yo - 1.5*(yo - yi)
	yf = yo + 1.5*(yf - yo)
	dx = xf - xi
	dy = yf - yi
	if (x < xi or x > xf or y < yi or y > yf):
		return 0.0

	return 0.25*(1.0 + np.cos(2.0*np.pi*(x-xo)/dx))*(1.0 + np.cos(2.0*np.pi*(y-yo)/dy))

def vort_func(x,y):
	return -2.0*np.cos(x)*np.cos(y)

def Plot(x,y,dat,title,fname,zmin,zmax):
	levs = np.linspace(zmin,zmax,201,endpoint=True)
	#plt.contourf(x,y,dat,levs)
	plt.contourf(x,y,dat,101)
	plt.colorbar()
	plt.title(title)
	plt.savefig(fname)
	plt.clf()

def TestConservation(topo,quad,lx,ly,g,h,w,k):
	det = 0.5*lx/topo.nx*0.5*ly/topo.ny

	n = topo.n
	np1 = topo.n + 1
	mp1 = quad.n + 1

	edge = LagrangeEdge(n)
	Ejxi = np.zeros((mp1,n),dtype=np.float64)
	for j in np.arange(n):
		for i in np.arange(mp1):
			Ejxi[i,j] = edge.eval(quad.x[i],j)

	node = LagrangeNode(n)
	Njxi = np.zeros((mp1,np1),dtype=np.float64)
	for j in np.arange(np1):
		for i in np.arange(mp1):
			Njxi[i,j] = node.eval(quad.x[i],j)

	volume = 0.0
	potVort = 0.0
	potEnst = 0.0
	totEner = 0.0

	for ex in np.arange(topo.nx):
		for ey in np.arange(topo.ny):
			inds0 = topo.localToGlobal0(ex,ey)
			inds2 = topo.localToGlobal2(ex,ey)

			volume_e = 0.0
			potVort_e = 0.0
			potEnst_e = 0.0
			totEner_e = 0.0
			for jj in np.arange(mp1*mp1):
				jx = jj%mp1
				jy = jj/mp1
				xl = quad.x[jx]
				yl = quad.x[jy]
				wx = quad.w[jx]
				wy = quad.w[jy]
				wt = wx*wy

				wq = 0.0
				for ii in np.arange(np1*np1):
					ix = ii%np1
					iy = ii/np1
					wq = wq + w[inds0[ii]]*Njxi[jx,ix]*Njxi[jy,iy]

				hq = 0.0
				kq = 0.0
				for ii in np.arange(n*n):
					ix = ii%n
					iy = ii/n
					hq = hq + h[inds2[ii]]*Ejxi[jx,ix]*Ejxi[jy,iy]
					kq = kq + k[inds2[ii]]*Ejxi[jx,ix]*Ejxi[jy,iy]

				volume_e = volume_e + wt*hq
				potVort_e = potVort_e + wt*wq/hq
				potEnst_e = potEnst_e + wt*wq*wq/hq
				totEner_e = totEner_e + wt*(hq*kq + 0.5*g*hq*hq)

			volume = volume + volume_e
			potVort = potVort + potVort_e
			potEnst = potEnst + potEnst_e
			totEner = totEner + totEner_e

	return det*volume, det*potVort, det*potEnst, det*totEner

nx = 16
ny = 16
n = 3 # basis order
m = 6 # quadrature order
np1 = n+1
mp1 = m+1
nxn = nx*n
nyn = ny*n
nxm = nx*m
nym = ny*m
shift1Form = nxn*nyn
shiftQuad = nxm*nym

topo = Topo(nx,ny,n)
quad = GaussLobatto(m)
topo_q = Topo(nx,ny,m)

lx = 2.0*np.pi
ly = 2.0*np.pi
dx = lx/nx
dy = ly/ny

node = LagrangeNode(n)
Njxi = np.zeros((mp1,np1),dtype=np.float64)
for j in np.arange(np1):
	for i in np.arange(mp1):
		Njxi[i,j] = node.eval(quad.x[i],j)

edge = LagrangeEdge(n)
Ejxi = np.zeros((mp1,n),dtype=np.float64)
for j in np.arange(n):
	for i in np.arange(mp1):
		Ejxi[i,j] = edge.eval(quad.x[i],j)

p2d = np.zeros((nym,nxm),dtype=np.float64)
u2d = np.zeros((nym,nxm),dtype=np.float64)
v2d = np.zeros((nym,nxm),dtype=np.float64)
w2d = np.zeros((nym,nxm),dtype=np.float64)
k2d = np.zeros((nym,nxm),dtype=np.float64)
dk2d = np.zeros((nxm,nxm),dtype=np.float64)
wu2d = np.zeros((nxm,nxm),dtype=np.float64)

x = np.zeros(nxm)
y = np.zeros(nym)
px = np.zeros(nxm*nym,dtype=np.float64)
ux = np.zeros(2*nxm*nym,dtype=np.float64)

print 'initializing shallow water equation...'

for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		inds1x = topo.localToGlobal1x(ex,ey)
		inds1y = topo.localToGlobal1y(ex,ey) + shift1Form
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			x[i] = ex*dx + 0.5*dx*(quad.x[jj%mp1] + 1.0)
			y[j] = ey*dy + 0.5*dy*(quad.x[jj/mp1] + 1.0)
			#px[inds0[jj]] = 1.0 + 0.1*np.cos(x[i])*np.cos(y[j])
			#ux[inds0[jj]] = 0.0
			px[inds0[jj]] = 1.0
			ux[inds0[jj]] = 0.2 + 0.1*np.sin(y[j])

			u2d[j,i] = 0.2 + 0.1*np.sin(y[j])
			k2d[j,i] = 0.5*(0.2 + 0.1*np.sin(y[j]))*(0.2 + 0.1*np.sin(y[j]))
			dk2d[j,i] = +0.5*2.0*(0.2 + 0.1*np.sin(y[j]))*0.1*np.cos(y[j])
			wu2d[j,i] = -0.5*2.0*(0.2 + 0.1*np.sin(y[j]))*0.1*np.cos(y[j])

Plot(x,y,u2d,'vel x (initial)','ux_000.png',0.9,1.1)
Plot(x,y,k2d,'ke (initial)','ke_000.png',0.0,0.0)
Plot(x,y,dk2d,'grad ke (initial)','dkx_000.png',0.0,0.0)
Plot(x,y,wu2d,'wu (initial)','wux_000.png',0.0,0.0)

Mxto1 = Xto1(topo,quad).M
Mxto2 = Xto2(topo,quad).M
hi = Mxto2*px
ui = Mxto1*ux

M0 = Pmat(topo,quad).M
M1 = Umat(topo,quad).M
M2 = Wmat(topo,quad).M
M0inv = la.inv(M0)
M2inv = la.inv(M2)

D10 = BoundaryMat10(topo).M
D01 = D10.transpose()

M0invD01 = M0inv*D01
D = M0invD01*M1
wi = (2.0*nx/lx)*D*ui

WtQU = WtQUmat(topo,quad,ui).M
M2invWtQU = M2inv*WtQU
ki = 0.5*M2invWtQU*ui

dt = 0.4*dx/1.0/n
g = 1.0
sw = SWEqn(topo,quad,topo_q,lx,ly,g)
nsteps = 20

dt = 0.5*dt
nsteps = 2*nsteps

volC = np.zeros((nsteps),dtype=np.float64)
pvC = np.zeros((nsteps),dtype=np.float64)
peC = np.zeros((nsteps),dtype=np.float64)
teC = np.zeros((nsteps),dtype=np.float64)

vol0, pv0, pe0, te0 = TestConservation(topo,quad,lx,ly,g,hi,wi,ki)
print 'initial volume              %10.8e'%vol0
print 'initial potential vorticity %10.8e'%pv0
print 'initial potential enstrophy %10.8e'%pe0
print 'initial total energy        %10.8e'%te0

for step in np.arange(nsteps) + 1:
	print '\tsolving shallow water for time step %.3d'%step

	uf,hf,wi,ki = sw.solveRK2(ui,hi,dt)

	print '\tplotting height and velocity fields %.3d'%step

	for ey in np.arange(ny):
		for ex in np.arange(nx):
			inds0 = topo_q.localToGlobal0(ex,ey)
			inds2 = topo.localToGlobal2(ex,ey)
			for jj in np.arange(mp1*mp1):
				if jj%mp1 == m or jj/mp1 == m:
					continue
				i = inds0[jj]%nxm
				j = inds0[jj]/nxm
				ii = i%m
				jj = j%m
				p2d[j][i] = 0.0
				for kk in np.arange(n*n):
					p2d[j][i] = p2d[j][i] + hf[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]

	Plot(x,y,p2d,'p (numeric)','pres_%.3d'%step + '.png',-0.1,+0.1)

	for ey in np.arange(ny):
		for ex in np.arange(nx):
			inds0 = topo_q.localToGlobal0(ex,ey)
			inds1x = topo.localToGlobal1x(ex,ey)
			inds1y = topo.localToGlobal1y(ex,ey) + shift1Form
			for jj in np.arange(mp1*mp1):
				if jj%mp1 == m or jj/mp1 == m:
					continue
				i = inds0[jj]%nxm
				j = inds0[jj]/nxm
				ii = i%m
				jj = j%m
				u2d[j][i] = 0.0
				v2d[j][i] = 0.0
				for kk in np.arange(np1*n):
					u2d[j][i] = u2d[j][i] + uf[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
					v2d[j][i] = v2d[j][i] + uf[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

	Plot(x,y,u2d,'u (numeric)','velx_%.3d'%step + '.png',-1.0,+1.0)
	Plot(x,y,v2d,'v (numeric)','vely_%.3d'%step + '.png',-1.0,+1.0)

	for ey in np.arange(ny):
		for ex in np.arange(nx):
			inds0q = topo_q.localToGlobal0(ex,ey)
			inds0n = topo.localToGlobal0(ex,ey)
			for jj in np.arange(mp1*mp1):
				if jj%mp1 == m or jj/mp1 == m:
					continue
				i = inds0q[jj]%nxm
				j = inds0q[jj]/nxm
				ii = i%m
				jj = j%m
				w2d[j][i] = 0.0
				for kk in np.arange(np1*np1):
					w2d[j][i] = w2d[j][i] + wi[inds0n[kk]]*Njxi[ii,kk%np1]*Njxi[jj,kk/np1]

	Plot(x,y,w2d,'w (numeric)','vort_%.3d'%step + '.png',-0.1,+0.1)

	for ey in np.arange(ny):
		for ex in np.arange(nx):
			inds0 = topo_q.localToGlobal0(ex,ey)
			inds2 = topo.localToGlobal2(ex,ey)
			for jj in np.arange(mp1*mp1):
				if jj%mp1 == m or jj/mp1 == m:
					continue
				i = inds0[jj]%nxm
				j = inds0[jj]/nxm
				ii = i%m
				jj = j%m
				k2d[j][i] = 0.0
				for kk in np.arange(n*n):
					k2d[j][i] = k2d[j][i] + ki[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]

	Plot(x,y,k2d,'ke (numeric)','ke_%.3d'%step + '.png',0.0,+0.5)

	vol, pv, pe, te = TestConservation(topo,quad,lx,ly,g,hi,wi,ki)

	for ey in np.arange(ny):
		for ex in np.arange(nx):
			inds0q = topo_q.localToGlobal0(ex,ey)
			inds0 = topo.localToGlobal0(ex,ey)
			inds1x = topo.localToGlobal1x(ex,ey)
			for jj in np.arange(mp1*mp1):
				if jj%mp1 == m or jj/mp1 == m:
					continue
				i = inds0q[jj]%nxm
				j = inds0q[jj]/nxm
				ii = i%m
				jj = j%m

				wq = 0.0
				for kk in np.arange(np1*np1):
					wq = wq + wi[inds0[kk]]*Njxi[ii,kk%np1]*Njxi[jj,kk/np1]

				uq = 0.0
				for kk in np.arange(np1*n):
					uq = uq + ui[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]

				wu2d[j,i] = wq*uq

	Plot(x,y,wu2d,'wu (numeric)','wux_%.3d'%step + '.png',0.0,+0.5)

	print '\tvolume conservation:               %10.8e'%((vol - vol0)/vol0)
	print '\tpotential vorticity conservation:  %10.8e'%((pv - pv0))
	#print '\tpotential enstrophy conservation:  %10.8e'%((pe - pe0)/pe0)
	print '\tpotential enstrophy conservation:  %10.8e'%((pe - pe0))
	print '\ttotal energy conservation:         %10.8e'%((te - te0)/te0)

	volC[step-1] = (vol-vol0)/vol0
	pvC[step-1] = (pv-pv0)
	peC[step-1] = (pe-pe0)
	teC[step-1] = (te-te0)/te0

	print volC[:step]
	print pvC[:step]
	print peC[:step]
	print teC[:step]

	ui[:] = uf[:]
	hi[:] = hf[:]
