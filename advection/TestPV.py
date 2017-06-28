#!/usr/bin/env python

import sys
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

def psi(x,y):
	xo = x - np.pi
	yo = y - np.pi
	return np.exp(-1.0*(xo*xo + yo*yo))

def velx(x,y):
	xo = x - np.pi
	yo = y - np.pi
	return +2.0*yo*psi(x,y) 
def vely(x,y):
	xo = x - np.pi
	yo = y - np.pi
	return -2.0*xo*psi(x,y)

def pres(x,y):
	xo = x - np.pi
	yo = y - np.pi
	return 1.0 + 0.25*(np.cos(xo) + 1.0)*(np.cos(yo) + 1.0)

def vort(x,y):
	xo = x - np.pi
	yo = y - np.pi
	strm = psi(x,y)
	return 4.0*(xo*xo + yo*yo - 1.0)*strm

def Plot(x,y,dat,title,fname,zmin,zmax):
	if np.abs(zmin) > 0.01 or np.abs(zmax) > 0.01:
		levs = np.linspace(zmin,zmax,101,endpoint=True)
		plt.contourf(x,y,dat,levs)
	else:
		plt.contourf(x,y,dat,101)
	plt.colorbar()
	plt.title(title)
	plt.savefig(fname)
	plt.clf()

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

u2dn = np.zeros((nym,nxm),dtype=np.float64)
v2dn = np.zeros((nym,nxm),dtype=np.float64)
q2dn = np.zeros((nym,nxm),dtype=np.float64)
h2dn = np.zeros((nym,nxm),dtype=np.float64)
u2da = np.zeros((nym,nxm),dtype=np.float64)
v2da = np.zeros((nym,nxm),dtype=np.float64)
q2da = np.zeros((nym,nxm),dtype=np.float64)
h2da = np.zeros((nym,nxm),dtype=np.float64)

f = 1.0
g = 1.0
H = 1.0

x = np.zeros(nxm)
y = np.zeros(nym)
ux = np.zeros(2*nxm*nym,dtype=np.float64)
hx = np.zeros(nxm*nym,dtype=np.float64)

sw = SWEqn(topo,quad,topo_q,lx,ly,f,g,0.0)

print 'initializing primitive variables...'
for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			x[i] = ex*dx + 0.5*dx*(quad.x[jj%mp1] + 1.0)
			y[j] = ey*dy + 0.5*dy*(quad.x[jj/mp1] + 1.0)
			ux[inds0[jj]] = velx(x[i],y[j])
			ux[inds0[jj]+shiftQuad] = vely(x[i],y[j])
			hx[inds0[jj]] = pres(x[i],y[j])

Mxto2 = Xto2(topo,quad).M
Mxto1 = Xto1(topo,quad).M
ui = Mxto1*ux
hi = Mxto2*hx

qi = sw.diagnose_q(hi,ui)

for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		inds0n = topo.localToGlobal0(ex,ey)
		inds1x = topo.localToGlobal1x(ex,ey)
		inds1y = topo.localToGlobal1y(ex,ey)
		inds2 = topo.localToGlobal2(ex,ey)
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			ii = i%m
			jj = j%m

			q2dn[j][i] = 0.0
			for kk in np.arange(np1*np1):
				q2dn[j][i] = q2dn[j][i] + qi[inds0n[kk]]*Njxi[ii,kk%np1]*Njxi[jj,kk/np1]

			q2da[j][i] = (vort(x[i],y[j]) + 1.0)/pres(x[i],y[j])

			u2dn[j][i] = 0.0
			v2dn[j][i] = 0.0
			for kk in np.arange(np1*n):
				u2dn[j][i] = u2dn[j][i] + ui[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
				v2dn[j][i] = v2dn[j][i] + ui[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

			u2da[j][i] = velx(x[i],y[j])
			v2da[j][i] = vely(x[i],y[j])

			h2dn[j][i] = 0.0
			for kk in np.arange(n*n):
				h2dn[j][i] = h2dn[j][i] + hi[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]

			h2da[j][i] = pres(x[i],y[j])

			q2da[j][i] = (vort(x[i],y[j]) + 1.0)/pres(x[i],y[j])


Plot(x,y,q2dn,'q (numeric)','q_num.png',0.0,0.0)
Plot(x,y,q2da,'q (analytic)','q_ana.png',0.0,0.0)
Plot(x,y,u2dn,'u (numeric)','u_num.png',0.0,0.0)
Plot(x,y,u2da,'u (analytic)','u_ana.png',0.0,0.0)
Plot(x,y,v2dn,'v (numeric)','v_num.png',0.0,0.0)
Plot(x,y,v2da,'v (analytic)','v_ana.png',0.0,0.0)
Plot(x,y,h2dn,'h (numeric)','h_num.png',0.0,0.0)
Plot(x,y,h2da,'h (analytic)','h_ana.png',0.0,0.0)
