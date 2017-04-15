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

def velx_func(x,y):
	return np.cos(x)*np.sin(y)

def vely_func(x,y):
	return np.sin(x)*np.cos(y)

def pres_func(x,y):
	return np.sin(x)*np.sin(y)

def lied_func(x,y):
	return np.cos(2.0*x)*np.sin(y)*np.sin(y) + np.cos(2.0*y)*np.sin(x)*np.sin(x)

def Plot(x,y,dat,title,fname):
	plt.contourf(x,y,dat,100)
	plt.colorbar()
	plt.title(title)
	plt.savefig(fname)
	plt.clf()

nx = 14
ny = 7
#nx = 8
#ny = 4
n = 3
m = n
np1 = n+1
mp1 = m+1
nxn = nx*n
nyn = ny*n
nxm = nx*m
nym = ny*m
shift = nxm*nym

topo = Topo(nx,ny,n)
quad = GaussLobatto(m)
topo_q = Topo(nx,ny,m)

lx = 4.0*np.pi
ly = 2.0*np.pi
dx = lx/nx
dy = ly/ny

print 'plotting analytic fields......'

x = np.zeros(nxm)
y = np.zeros(nym)
px = np.zeros(nxm*nym,dtype=np.float64)
ux = np.zeros(2*nxm*nym,dtype=np.float64)
p2d = np.zeros((nym,nxm),dtype=np.float64)
u2d = np.zeros((nym,nxm),dtype=np.float64)
v2d = np.zeros((nym,nxm),dtype=np.float64)
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
			px[inds0[jj]] = pres_func(x[i],y[j])
			ux[inds0[jj]] = velx_func(x[i],y[j])
			ux[inds0[jj]+shift] = vely_func(x[i],y[j])
			p2d[j][i] = px[inds0[jj]]
			u2d[j][i] = ux[inds0[jj]]
			v2d[j][i] = ux[inds0[jj]+shift]

Plot(x,y,p2d,'p (analytic)','pres_a.png')
Plot(x,y,u2d,'u (analytic)','velx_a.png')
Plot(x,y,v2d,'v (analytic)','vely_a.png')

print 'mapping to basis..............'

Mxto1 = Xto1(topo,quad).M
Mxto2 = Xto2(topo,quad).M
u = Mxto1*ux
p = Mxto2*px

print 'plotting p 2 forms............'

node = LagrangeNode(n)
edge = LagrangeEdge(n)
for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		inds2 = topo.localToGlobal2(ex,ey)
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			xl = quad.x[i%m]
			yl = quad.x[j%m]
			p2d[j][i] = 0.0
			for kk in np.arange(n*n):
				p2d[j][i] = p2d[j][i] + p[inds2[kk]]*edge.eval(xl,kk%n)*edge.eval(yl,kk/n)

Plot(x,y,p2d,'p (numeric)','pres_n.png')

print 'plotting u 1 forms............'

for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		inds1x = topo.localToGlobal1x(ex,ey)
		inds1y = topo.localToGlobal1y(ex,ey)
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			xl = quad.x[i%m]
			yl = quad.x[j%m]
			u2d[j][i] = 0.0
			v2d[j][i] = 0.0
			for kk in np.arange(np1*n):
				u2d[j][i] = u2d[j][i] + u[inds1x[kk]]*node.eval(xl,kk%np1)*edge.eval(yl,kk/np1)
				v2d[j][i] = v2d[j][i] + u[inds1y[kk]]*edge.eval(xl,kk%n)*node.eval(yl,kk/n)

Plot(x,y,u2d,'u (numeric)','velx_n.png')
Plot(x,y,v2d,'v (numeric)','vely_n.png')

print 'initializing lie derivative...'

lie = LieDeriv(topo,quad,lx,ly)

print 'solving lie derivative........'

duq, uq = lie.assemble(u,p)

print 'plotting output fields........'

dq2d_a = np.zeros((nym,nxm),dtype=np.float64)
dq2d_n = np.zeros((nym,nxm),dtype=np.float64)
qx2d_n = np.zeros((nym,nxm),dtype=np.float64)
qx2d_a = np.zeros((nym,nxm),dtype=np.float64)
qy2d_n = np.zeros((nym,nxm),dtype=np.float64)
qy2d_a = np.zeros((nym,nxm),dtype=np.float64)

for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		inds2 = topo.localToGlobal2(ex,ey)
		inds1x = topo.localToGlobal1x(ex,ey)
		inds1y = topo.localToGlobal1y(ex,ey)
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			xl = quad.x[i%m]
			yl = quad.x[j%m]
			dq2d_a[j][i] = lied_func(x[i],y[j])
			qx2d_a[j][i] = pres_func(x[i],y[j])*velx_func(x[i],y[j])
			qy2d_a[j][i] = pres_func(x[i],y[j])*vely_func(x[i],y[j])
			dq2d_n[j][i] = 0.0
			for kk in np.arange(n*n):
				dq2d_n[j][i] = dq2d_n[j][i] + duq[inds2[kk]]*edge.eval(xl,kk%n)*edge.eval(yl,kk/n)
			qx2d_n[j][i] = 0.0
			qy2d_n[j][i] = 0.0
			for kk in np.arange(n*np1):
				qx2d_n[j][i] = qx2d_n[j][i] + uq[inds1x[kk]]*node.eval(xl,kk%np1)*edge.eval(yl,kk/np1)
				qy2d_n[j][i] = qy2d_n[j][i] + uq[inds1y[kk]]*edge.eval(xl,kk%n)*node.eval(yl,kk/n)
			
Plot(x,y,dq2d_a,'dq (analytic)','lied_a.png' )
Plot(x,y,dq2d_n,'dq (numeric)' ,'lied_n.png' )
Plot(x,y,qx2d_a,'qx (analytic)','lieqx_a.png')
Plot(x,y,qx2d_n,'qx (numeric)' ,'lieqx_n.png')
Plot(x,y,qy2d_a,'qy (analytic)','lieqy_a.png')
Plot(x,y,qy2d_n,'qy (numeric)' ,'lieqy_n.png')
