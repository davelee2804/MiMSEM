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

def vort(x,y):
	xo = x - np.pi
	yo = y - np.pi
	return 0.25*(np.cos(xo) + 1.0)*(np.cos(yo) + 1.0)

def dqdx(x,y):
	xo = x - np.pi
	yo = y - np.pi
	return -0.25*np.sin(xo)*(np.cos(yo) + 1.0)

def dqdy(x,y):
	xo = x - np.pi
	yo = y - np.pi
	return -0.25*(np.cos(xo) + 1.0)*np.sin(yo)

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

nx = 20
ny = 20
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

sw = SWEqn(topo,quad,topo,lx,ly,f,g,+1.0)

print 'initializing primitive variables...'
for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo.localToGlobal0(ex,ey)
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			x[i] = ex*dx + 0.5*dx*(quad.x[jj%mp1] + 1.0)
			y[j] = ey*dy + 0.5*dy*(quad.x[jj/mp1] + 1.0)
			hx[inds0[jj]] = vort(x[i],y[j])

Mxto2 = Xto2(topo,quad).M
hi = Mxto2*hx

dh = sw.M1inv*sw.D12M2*hi

for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo.localToGlobal0(ex,ey)
		inds1x = topo.localToGlobal1x(ex,ey)
		inds1y = topo.localToGlobal1y(ex,ey)
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			ii = i%m
			jj = j%m

			u2dn[j][i] = 0.0
			v2dn[j][i] = 0.0
			for kk in np.arange(np1*n):
				u2dn[j][i] = u2dn[j][i] + dh[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
				v2dn[j][i] = v2dn[j][i] + dh[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

			u2da[j][i] = dqdx(x[i],y[j])
			v2da[j][i] = dqdy(x[i],y[j])

Plot(x,y,u2dn,'dhx (numeric)','dhx_num.png',0.0,0.0)
Plot(x,y,v2dn,'dhy (numeric)','dhy_num.png',0.0,0.0)
Plot(x,y,u2da,'dhx (analytic)','dhx_ana.png',0.0,0.0)
Plot(x,y,v2da,'dhy (analytic)','dhy_ana.png',0.0,0.0)
