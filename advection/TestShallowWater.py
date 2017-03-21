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

def WriteMat(name, M_sparse, isInt):
    M = M_sparse.toarray()
    nr = M.shape[0]
    nc = M.shape[1]

    outfile = open(name, 'w')
    for r in np.arange(nr):
        for c in np.arange(nc):
			if isInt:
				outfile.write('%.2d'%M[r][c] + '\t')
			else:
				outfile.write('%2.3e'%M[r][c] + '\t')

        outfile.write('\n')

    outfile.close()

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
#quad = GaussLegendre(m)
topo_q = Topo(nx,ny,m)

lx = 2.0*np.pi
ly = 2.0*np.pi
dx = lx/nx
dy = ly/ny

print 'initializing analytic fields..........'

x = np.zeros(nxm)
y = np.zeros(nym)
px = np.zeros(nxm*nym,dtype=np.float64)
ux = np.zeros(2*nxm*nym,dtype=np.float64)
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
			ux[inds0[jj]+shiftQuad] = vely_func(x[i],y[j])

print 'mapping to basis......................'

Mxto1 = Xto1(topo,quad).M
Mxto2 = Xto2(topo,quad).M
u = Mxto1*ux
p = Mxto2*px

print 'testing w 0 form solve................'

M1 = Umat(topo,quad).M
M0 = Pmat(topo,quad).M
M0inv = la.inv(M0)
D10 = BoundaryMat10(topo).M
D01 = D10.transpose()
M0invD01 = M0inv*D01
D = M0invD01*M1
Utn = UNormToTang(topo,quad)
ut = Utn.apply(u)
w = (2.0*nx/lx)*D*u

print 'plotting *u 1 forms...................'

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
				u2d[j][i] = u2d[j][i] + ut[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
				v2d[j][i] = v2d[j][i] + ut[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

Plot(x,y,u2d,'*u (numeric)','velx_t_n.png',-1.0,+1.0)
Plot(x,y,v2d,'*v (numeric)','vely_t_n.png',-1.0,+1.0)

print 'plotting w 0 forms....................'

w2da = np.zeros((nym,nxm),dtype=np.float64)
w2dn = np.zeros((nym,nxm),dtype=np.float64)
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
			w2da[j][i] = vort_func(x[i],y[j])
			w2dn[j][i] = 0.0
			for kk in np.arange(np1*np1):
				w2dn[j][i] = w2dn[j][i] + w[inds0n[kk]]*Njxi[ii,kk%np1]*Njxi[jj,kk/np1]

Plot(x,y,w2da,'w (analytic)','vort_a.png',-2.0,+2.0)
Plot(x,y,w2dn,'w (numeric)','vort_n.png',-2.0,+2.0)

print 'plotting q 2 forms....................'

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
				p2d[j][i] = p2d[j][i] + p[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]

Plot(x,y,p2d,'p (numeric)','pres_n.png',-0.01,+1.0)

print 'plotting u 1 forms....................'

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
				u2d[j][i] = u2d[j][i] + u[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
				v2d[j][i] = v2d[j][i] + u[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

Plot(x,y,u2d,'u (numeric)','velx_n.png',-1.0,+1.0)
Plot(x,y,v2d,'v (numeric)','vely_n.png',-1.0,+1.0)

print 'plotting ke 2 form....................'

ke2dn = np.zeros((nym,nxm),dtype=np.float64)
ke2da = np.zeros((nym,nxm),dtype=np.float64)

M2 = Wmat(topo,quad).M
M2inv = la.inv(M2)
WtQU = WtQUmat(topo,quad,u).M
M2invWtQU = M2inv*WtQU
un2 = M2invWtQU*u
ke2 = 0.5*un2

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
			ke2da[j][i] = 0.5*(velx_func(x[i],y[j])*velx_func(x[i],y[j]) + vely_func(x[i],y[j])*vely_func(x[i],y[j]))
			ke2dn[j][i] = 0.0
			for kk in np.arange(n*n):
				ke2dn[j][i] = ke2dn[j][i] + ke2[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]

Plot(x,y,ke2dn,'ke (numeric)','ke_n.png',0.0,+0.5)
Plot(x,y,ke2da,'ke (analytic)','ke_a.png',0.0,+0.5)

print 'plotting vorticity-velocity 1 forms...'

uw2dn = np.zeros((nym,nxm),dtype=np.float64)
uw2da = np.zeros((nym,nxm),dtype=np.float64)

M1w = RotationalMat(topo,quad,w).M
M1inv = la.inv(M1)
UW = M1inv*M1w
ut[:shift1Form] = -1.0*ut[:shift1Form]
wu = UW*ut

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
				u2d[j][i] = u2d[j][i] + wu[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
				v2d[j][i] = v2d[j][i] + wu[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

Plot(x,y,u2d,'wu (numeric)','wux_n.png',-1.0,+1.0)
Plot(x,y,v2d,'wv (numeric)','wuy_n.png',-1.0,+1.0)

for ey in np.arange(ny):
	for ex in np.arange(nx):
		inds0 = topo_q.localToGlobal0(ex,ey)
		for jj in np.arange(mp1*mp1):
			if jj%mp1 == m or jj/mp1 == m:
				continue
			i = inds0[jj]%nxm
			j = inds0[jj]/nxm
			u2d[j][i] = -vely_func(x[i],y[j])*vort_func(x[i],y[j])
			v2d[j][i] = +velx_func(x[i],y[j])*vort_func(x[i],y[j])

Plot(x,y,u2d,'wu (analytic)','wux_a.png',-1.0,+1.0)
Plot(x,y,v2d,'wv (analytic)','wuy_a.png',-1.0,+1.0)

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
			px[inds0[jj]] = 0.1*np.cos(x[i])*np.cos(y[j])

p = Mxto2*px
u[:] = 0.0

dt = 0.4*dx/1.0/n
sw = SWEqn(topo,quad,topo_q,lx,ly,1.0)
nsteps = 100
ui = u
hi = p
for step in np.arange(nsteps) + 1:
	print '\tsolving shallow water for time step %.3d'%step

	uf,hf = sw.solveRK2(ui,hi,dt)

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

	ui[:] = uf[:]
	hi[:] = hf[:]
