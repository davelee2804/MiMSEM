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

def velx_func(x,y):
	#return +1.0*np.cos(x)*np.sin(y)
	return +1.0*np.sin(y)

def vely_func(x,y):
	#return -1.0*np.sin(x)*np.cos(y)
	return -1.0*np.sin(x)

def pres_func(x,y):
	xi = 2.0*np.pi/8.0 + 2.0*np.pi/4.0
	xf = 6.0*np.pi/8.0 + 2.0*np.pi/4.0
	#yi = 2.0*np.pi/8.0 + 2.0*np.pi/16.0
	#yf = 6.0*np.pi/8.0 + 2.0*np.pi/16.0
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

def Plot(x,y,dat,title,fname,tmin,tmax):
	levs = np.linspace(tmin,tmax,201,endpoint=True)
	plt.contourf(x,y,dat,levs)
	plt.colorbar()
	plt.title(title)
	plt.savefig(fname)
	plt.clf()

def Integrate2Form(topo,quad,lx,ly,q):
	det = 0.5*lx/topo.nx*0.5*ly/topo.ny
	edge = LagrangeEdge(topo.n)

	np1 = quad.n + 1

	qt = 0.0
	for ex in np.arange(topo.nx):
		for ey in np.arange(topo.ny):
			inds2 = topo.localToGlobal2(ex,ey)

			qe = 0.0
			for jj in np.arange(np1*np1):
				jx = jj%np1
				jy = jj/np1
				xl = quad.x[jx]
				yl = quad.x[jy]
				wx = quad.w[jx]
				wy = quad.w[jy]
				wt = wx*wy

				qi = 0.0
				for ii in np.arange(topo.n*topo.n):
					ix = ii%topo.n
					iy = ii/topo.n
					qi = qi + q[inds2[ii]]*edge.eval(xl,ix)*edge.eval(yl,iy)

				qe = qe + wt*qi

			qt = qt + qe

	return det*qt

nx = 20
ny = 20
n = 3
m = 3
np1 = n+1
mp1 = m+1
nxn = nx*n
nyn = ny*n
nxm = nx*m
nym = ny*m
shift = nxm*nym

topo = Topo(nx,ny,n)
quad = GaussLobatto(m)
#quad = GaussLegendre(m)
topo_q = Topo(nx,ny,m)

lx = 2.0*np.pi
ly = 2.0*np.pi
dx = lx/nx
dy = ly/ny

print 'initializing analytic fields......'

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

print 'mapping to basis..................'

Mxto1 = Xto1(topo,quad).M
Mxto2 = Xto2(topo,quad).M
u = Mxto1*ux
p = Mxto2*px

print 'plotting q 2 forms................'

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

Plot(x,y,p2d,'p (numeric)','pres_n.png',-0.01,1.01)

print 'plotting u 1 forms................'

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

Plot(x,y,u2d,'u (numeric)','velx_n.png',-1.0,+1.0)
Plot(x,y,v2d,'v (numeric)','vely_n.png',-1.0,+1.0)

print 'initializing advection equation...'

dt = 0.4*dx/1.0/n
qi = p
adv = AdvectionEqn(topo,quad,lx,ly)

intp0 = Integrate2Form(topo,quad,lx,ly,p)
print 'initial mass: %12.8e'%intp0

for step in np.arange(60) + 1:
	print '\tsolving advection equation for step: %.3u'%step

	if step == 1:
		qf = adv.solveRK2(u,qi,dt,True)
	else:
		qf = adv.solveRK2(u,qi,dt,False)

	print '\tplotting q 2 forms for step:         %.3u'%step
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
					p2d[j][i] = p2d[j][i] + qf[inds2[kk]]*edge.eval(xl,kk%n)*edge.eval(yl,kk/n)

	intpf = Integrate2Form(topo,quad,lx,ly,qf)
	print '\tmass conservation: %16.14e'%((intpf - intp0)/intp0)

	Plot(x,y,p2d,'p (numeric)','pres_n_%.4u'%step + '.png',-0.01,+1.01)

	qi[:] = qf[:]

