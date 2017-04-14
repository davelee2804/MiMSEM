#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from BoundaryMat import *
from Mats2D import *
from Assembly import *
from Proj import *
from ShallowWaterEqn import *

def psi1(x,y):
	xl = x - np.pi - np.pi/3.0
	yo = y - np.pi
	return np.exp(-4.0*(xl*xl + yo*yo))

def psi2(x,y):
	xr = x - np.pi + np.pi/3.0
	yo = y - np.pi
	return np.exp(-4.0*(xr*xr + yo*yo))

def strm_func(x,y):
	return psi1(x,y) + psi2(x,y)

def velx_func(x,y):
	yo = y - np.pi
	return +8.0*yo*psi1(x,y) + 8.0*yo*psi2(x,y)

def vely_func(x,y):
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	return -8.0*xl*psi1(x,y) - 8.0*xr*psi2(x,y)

def pres_func(x,y):
	H = 1.0
	f = 10.0
	g = 10.0
	return (f/g)*strm_func(x,y) + H

def dpdx_func(x,y):
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	return -8.0*xl*psi1(x,y) - 8.0*xr*psi2(x,y)

def dpdy_func(x,y):
	yo = y - np.pi
	return -8.0*yo*psi1(x,y) - 8.0*yo*psi2(x,y)

def d2pdxx_func(x,y):
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	return - 8.0*psi1(x,y) - 8.0*psi2(x,y) + 64.0*xl*xl*psi1(x,y) + 64.0*xr*xr*psi2(x,y)

def d2pdyy_func(x,y):
	yo = y - np.pi
	return - 8.0*psi1(x,y) - 8.0*psi2(x,y) + 64.0*yo*yo*psi1(x,y) + 64.0*yo*yo*psi2(x,y)

def vort_func(x,y):
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	yo = y - np.pi
	w1 = 16.0*(4.0*(xl*xl + yo*yo) - 1.0)*psi1(x,y)
	w2 = 16.0*(4.0*(xr*xr + yo*yo) - 1.0)*psi2(x,y)
	return (w1 + w2 + 10.0)/pres_func(x,y)

def momx_func(x,y):
	return pres_func(x,y)*velx_func(x,y)

def momy_func(x,y):
	return pres_func(x,y)*vely_func(x,y)

def Plot(x,y,dat,title,fname,zmin,zmax):
	levs = np.linspace(zmin,zmax,201,endpoint=True)
	plt.contourf(x,y,dat,101)
	plt.colorbar()
	plt.title(title)
	plt.savefig(fname)
	plt.clf()

def TestConservation(topo,quad,lx,ly,f,g,h,u,q,D01):
	det = 0.5*lx/topo.nx*0.5*ly/topo.ny
	shift1Form = topo.nx*topo.ny*topo.n*topo.n

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

	w = D01*u

	volume = 0.0
	potVort = 0.0
	potEnst = 0.0
	totEner = 0.0

	for ex in np.arange(topo.nx):
		for ey in np.arange(topo.ny):
			inds0 = topo.localToGlobal0(ex,ey)
			inds2 = topo.localToGlobal2(ex,ey)
			inds1x = topo.localToGlobal1x(ex,ey)
			inds1y = topo.localToGlobal1x(ex,ey) + shift1Form

			volume_e = 0.0
			potVort_e = 0.0
			potEnst_e = 0.0
			totEner_e = 0.0
			for jj in np.arange(mp1*mp1):
				jx = jj%mp1
				jy = jj/mp1
				wx = quad.w[jx]
				wy = quad.w[jy]
				wt = wx*wy

				qq = 0.0
				wq = 0.0
				for ii in np.arange(np1*np1):
					ix = ii%np1
					iy = ii/np1
					qq = qq + q[inds0[ii]]*Njxi[jx,ix]*Njxi[jy,iy]
					wq = wq + w[inds0[ii]]*Njxi[jx,ix]*Njxi[jy,iy]

				hq = 0.0
				for ii in np.arange(n*n):
					ix = ii%n
					iy = ii/n
					hq = hq + h[inds2[ii]]*Ejxi[jx,ix]*Ejxi[jy,iy]

				uq = 0.0
				for ii in np.arange(n*np1):
					ix = ii%np1
					iy = ii/np1
					uq = uq + u[inds1x[ii]]*Njxi[jx,ix]*Ejxi[jy,iy]

				vq = 0.0
				for ii in np.arange(n*np1):
					ix = ii%n
					iy = ii/n
					vq = vq + u[inds1y[ii]]*Ejxi[jx,ix]*Njxi[jy,iy]

				volume_e = volume_e + wt*hq
				#potVort_e = potVort_e + wt*(qq*hq-f)
				potVort_e = potVort_e + wt*wq
				potEnst_e = potEnst_e + 0.5*wt*qq*qq*hq
				totEner_e = totEner_e + 0.5*wt*(hq*(uq*uq + vq*vq) + g*hq*hq)

			volume = volume + volume_e
			potVort = potVort + potVort_e
			potEnst = potEnst + potEnst_e
			totEner = totEner + totEner_e

	return det*volume, det*potVort, det*potEnst, det*totEner

nx = 16
ny = 16
n = 3 # basis order
m = 3 # quadrature order
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
q2d = np.zeros((nym,nxm),dtype=np.float64)
h2d = np.zeros((nym,nxm),dtype=np.float64)

dt = 0.20*dx/2.0/n
f = 10.0
g = 10.0
nsteps = 10

x = np.zeros(nxm)
y = np.zeros(nym)
px = np.zeros(nxm*nym,dtype=np.float64)
ux = np.zeros(2*nxm*nym,dtype=np.float64)
wx = np.zeros(nxm*nym,dtype=np.float64)
fx = f*np.ones(nxm*nym,dtype=np.float64)

sw = SWEqn(topo,quad,topo_q,lx,ly,fx,g)

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
			ux[inds0[jj]] = velx_func(x[i],y[j])
			ux[inds0[jj]+shiftQuad] = vely_func(x[i],y[j])
			px[inds0[jj]] = pres_func(x[i],y[j])

Mxto1 = Xto1(topo,quad).M
ui = Mxto1*ux
Mxto0 = Xto0(topo,quad).M
Mxto2 = Xto2(topo,quad).M
hi = Mxto2*px
qi = sw.diagnose_q(hi,ui)

for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		inds0n = topo.localToGlobal0(ex,ey)
		inds1x = topo.localToGlobal1x(ex,ey)
		inds1y = topo.localToGlobal1y(ex,ey) + shift1Form
		inds2 = topo.localToGlobal2(ex,ey)
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
				u2d[j][i] = u2d[j][i] + ui[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
				v2d[j][i] = v2d[j][i] + ui[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]
			h2d[j][i] = 0.0
			for kk in np.arange(n*n):
				h2d[j][i] = h2d[j][i] + hi[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]
			q2d[j][i] = 0.0
			for kk in np.arange(np1*np1):
				q2d[j][i] = q2d[j][i] + qi[inds0n[kk]]*Njxi[ii,kk%np1]*Njxi[jj,kk/np1]

Plot(x,y,h2d,'h (initial)','pres_000.png',0.0,0.0)
Plot(x,y,u2d,'u (initial)','velx_000.png',0.0,0.0)
Plot(x,y,v2d,'v (initial)','vely_000.png',0.0,0.0)
Plot(x,y,q2d,'w (initial)','vort_000.png',0.0,0.0)

for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		inds0n = topo.localToGlobal0(ex,ey)
		inds1x = topo.localToGlobal1x(ex,ey)
		inds1y = topo.localToGlobal1y(ex,ey) + shift1Form
		inds2 = topo.localToGlobal2(ex,ey)
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			ii = i%m
			jj = j%m
			q2d[j][i] = vort_func(x[i],y[j])
			u2d[j][i] = momx_func(x[i],y[j])
			v2d[j][i] = momy_func(x[i],y[j])

Plot(x,y,q2d,'w (test)','vort_tst.png',0.0,0.0)
Plot(x,y,u2d,'hu (test)','hu_tst.png',0.0,0.0)
Plot(x,y,v2d,'hv (test)','hv_tst.png',0.0,0.0)

Fi = sw.diagnose_F(hi,ui)
for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		inds0n = topo.localToGlobal0(ex,ey)
		inds1x = topo.localToGlobal1x(ex,ey)
		inds1y = topo.localToGlobal1y(ex,ey) + shift1Form
		inds2 = topo.localToGlobal2(ex,ey)
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
				u2d[j][i] = u2d[j][i] + Fi[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
				v2d[j][i] = v2d[j][i] + Fi[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

Plot(x,y,u2d,'hu (numeric)','hu_num.png',0.0,0.0)
Plot(x,y,v2d,'hv (numeric)','hv_num.png',0.0,0.0)

print 'testing boundary matrix........'

for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			ux[inds0[jj]] = dpdx_func(x[i],y[j])
			ux[inds0[jj]+shiftQuad] = dpdy_func(x[i],y[j])

u2 = Mxto1*ux
D21 = BoundaryMat(topo).M
D12 = -1.0*D21.transpose()
du = (2.0*nx/lx)*D21*u2

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
			h2d[j][i] = 0.0
			for kk in np.arange(n*n):
				h2d[j][i] = h2d[j][i] + du[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]

Plot(x,y,h2d,'du (numeric)','du_tst.png',0.0,0.0)

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
			h2d[j][i] = d2pdxx_func(x[i],y[j]) + d2pdyy_func(x[i],y[j])

Plot(x,y,h2d,'du (analytic)','du_ana.png',0.0,0.0)

M1 = Umat(topo,quad).M
M2 = Wmat(topo,quad).M
M1inv = la.inv(M1)
dp = (2.0*nx/lx)*M1inv*D12*M2*hi

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
				u2d[j][i] = u2d[j][i] + dp[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
				v2d[j][i] = v2d[j][i] + dp[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

Plot(x,y,u2d,'dpx (numeric)','dpx_num.png',0.0,0.0)
Plot(x,y,v2d,'dpy (numeric)','dpy_num.png',0.0,0.0)

for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			u2d[j][i] = dpdx_func(x[i],y[j])
			v2d[j][i] = dpdy_func(x[i],y[j])

Plot(x,y,u2d,'dpx (analytic)','dpx_tst.png',0.0,0.0)
Plot(x,y,v2d,'dpy (analytic)','dpy_tst.png',0.0,0.0)

dt = 1.0*dt
nsteps = 1*nsteps

volC = np.zeros((nsteps),dtype=np.float64)
pvC = np.zeros((nsteps),dtype=np.float64)
peC = np.zeros((nsteps),dtype=np.float64)
teC = np.zeros((nsteps),dtype=np.float64)

M0 = Pmat(topo,quad).M
M1 = Umat(topo,quad).M
M0inv = la.inv(M0)
D10 = BoundaryMat10(topo).M
D01 = D10.transpose()
Curl = M0inv*D01*M1
vol0, pv0, pe0, te0 = TestConservation(topo,quad,lx,ly,f,g,hi,ui,qi,Curl)
print 'initial volume              %10.8e'%vol0
print 'initial vorticity           %10.8e'%pv0
print 'initial potential enstrophy %10.8e'%pe0
print 'initial total energy        %10.8e'%te0

for step in np.arange(nsteps) + 1:
	print '\tsolving shallow water for time step %.3d'%step

	hf,uf = sw.solveRK2(hi,ui,dt)
	qf = sw.diagnose_q(hf,uf)

	print '\tplotting height and velocity fields %.3d'%step

	for ey in np.arange(ny):
		for ex in np.arange(nx):
			inds0q = topo_q.localToGlobal0(ex,ey)
			inds0 = topo.localToGlobal0(ex,ey)
			inds1x = topo.localToGlobal1x(ex,ey)
			inds1y = topo.localToGlobal1y(ex,ey) + shift1Form
			inds2 = topo.localToGlobal2(ex,ey)
			for jj in np.arange(mp1*mp1):
				if jj%mp1 == m or jj/mp1 == m:
					continue
				i = inds0q[jj]%nxm
				j = inds0q[jj]/nxm
				ii = i%m
				jj = j%m

				q2d[j][i] = 0.0
				for kk in np.arange(np1*np1):
					q2d[j][i] = q2d[j][i] + qf[inds0[kk]]*Njxi[ii,kk%np1]*Njxi[jj,kk/np1]

				u2d[j][i] = 0.0
				v2d[j][i] = 0.0
				for kk in np.arange(np1*n):
					u2d[j][i] = u2d[j][i] + uf[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
					v2d[j][i] = v2d[j][i] + uf[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

				p2d[j][i] = 0.0
				for kk in np.arange(n*n):
					p2d[j][i] = p2d[j][i] + hf[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]

	Plot(x,y,q2d,'w (numeric)','vort_%.3d'%step + '.png',0.0,0.0)
	Plot(x,y,u2d,'u (numeric)','velx_%.3d'%step + '.png',0.0,0.0)
	Plot(x,y,v2d,'v (numeric)','vely_%.3d'%step + '.png',0.0,0.0)
	Plot(x,y,p2d,'h (numeric)','pres_%.3d'%step + '.png',0.0,0.0)

	vol, pv, pe, te = TestConservation(topo,quad,lx,ly,f,g,hf,uf,qf,Curl)

	print '\tvolume conservation:               %10.8e'%((vol - vol0)/vol0)
	print '\tvorticity conservation:            %10.8e'%(pv - pv0)
	print '\tpotential enstrophy conservation:  %10.8e'%((pe - pe0)/pe0)
	print '\ttotal energy conservation:         %10.8e'%((te - te0)/te0)

	volC[step-1] = (vol-vol0)/vol0
	pvC[step-1] = (pv-pv0)
	peC[step-1] = (pe-pe0)/pe0
	teC[step-1] = (te-te0)/te0

	print repr(volC[:step])
	print repr(pvC[:step])
	print repr(peC[:step])
	print repr(teC[:step])

	hi[:] = hf[:]
	ui[:] = uf[:]
