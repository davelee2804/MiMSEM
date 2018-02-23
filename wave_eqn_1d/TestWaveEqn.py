#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from BoundaryMat import *
from Mats1D import *
from Assembly import *
from Proj import *
from WaveEqn import *
from WaveEqn_EEC import *
from WaveEqn_StdSEM import *

# dispersion relation: w = k(gH)**0.5; c_p = (gH)**0.5
# for initial depth: h = A.cos(k.x - w.t)
# initial u is: u = (g/H)**0.5.A.cos(k.x - w.t)

def velx_func(x,u):
	u[:] = 0.0

def pres_func(x,h):
	for ii in np.arange(x.shape[0]):
		if x[ii] < 0.5:
			h[ii] = 0.5 + 0.5*np.tanh(20.0*(x[ii] - 0.4))
		else:
			h[ii] = 0.5 + 0.5*np.tanh(20.0*(0.6 - x[ii]))

def plot(h,u,x,topo,topo_q,N,E,step,ho,uo,hc,uc,output_dir):
	ue = np.zeros(topo.n+1,dtype=np.float64)
	he = np.zeros(topo.n,dtype=np.float64)
	uoe = np.zeros(topo.n+1,dtype=np.float64)
	hoe = np.zeros(topo.n,dtype=np.float64)

	ux = np.zeros(topo_q.n*topo.nx,dtype=np.float64)
	hx = np.zeros(topo_q.n*topo.nx,dtype=np.float64)
	uox = np.zeros(topo_q.n*topo.nx,dtype=np.float64)
	hox = np.zeros(topo_q.n*topo.nx,dtype=np.float64)

	for ex in np.arange(topo.nx):
		inds0 = topo.localToGlobal0(ex)
		for ii in np.arange(topo.n+1):
			ue[ii] = u[inds0[ii]]
			uoe[ii] = uo[inds0[ii]]

		inds1 = topo.localToGlobal1(ex)
		for ii in np.arange(topo.n):
			he[ii] = h[inds1[ii]]
			hoe[ii] = ho[inds1[ii]]

		inds0q = topo_q.localToGlobal0(ex)
		for ii in np.arange(topo_q.n+1):
			if ii == 0 or ii == topo_q.n:
				a = 0.5
			else:
				a = 1.0
			kk = inds0q[ii]

			for jj in np.arange(topo.n+1):
				ux[kk] = ux[kk] + a*N[ii,jj]*ue[jj]
				uox[kk] = uox[kk] + a*N[ii,jj]*uoe[jj]

			for jj in np.arange(topo.n):
				hx[kk] = hx[kk] + a*E[ii,jj]*he[jj]
				hox[kk] = hox[kk] + a*E[ii,jj]*hoe[jj]

	uc[step,:] = ux[:]
	hc[step,:] = hx[:]

	plt.plot(x,uox,'ro')
	plt.plot(x,hox,'go')
	plt.plot(x,ux,'r-+')
	plt.plot(x,hx,'g-+')
	plt.ylim([-1.6,+1.6])
	plt.savefig(output_dir+'/wave_mim_%.4d'%step + '.png')
	plt.clf()

def energy(hi,ui,g,H,topo,quad,N,E,det):
	en = 0.0
	for ei in np.arange(topo.nx):
		inds0 = topo.localToGlobal0(ei)
		inds1 = topo.localToGlobal1(ei)
		for qi in np.arange(quad.n+1):
			hq = 0.0
			uq = 0.0

			for i in np.arange(topo.n + 1):
				uq = uq + N[qi,i]*ui[inds0[i]]

			for i in np.arange(topo.n):
				hq = hq + E[qi,i]*hi[inds1[i]]

			en = en + det*quad.w[qi]*(g*hq*hq + H*uq*uq)

	return en

eec = True

nx = 4
n = 6 # basis order
m = 6 # quadrature order
np1 = n+1
mp1 = m+1
nxn = nx*n
nxm = nx*m

topo = Topo(nx,n)
quad = GaussLobatto(m)
topo_q = Topo(nx,m)

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

lx = 1.0
dx = lx/nx
x = np.zeros(m*nx)
for i in np.arange(nx):
    x[i*m:(i+1)*m] = i*dx + (quad.x[:m] + 1.0)*0.5*dx

det = 0.5*lx/topo.nx

g = 10.0
H = 1.6
k = 2.0*np.pi
omega = k*np.sqrt(g*H)
po = 0.5
uo = -po*np.sqrt(g/H)
ux = uo*np.cos(k*x)
hx = po*np.cos(k*x)
velx_func(x,ux)
pres_func(x,hx)

time = lx/np.sqrt(g*H)
#nsteps = 200
nsteps = 800
dt = time/nsteps
print 'max step: %f'%(lx/(nx)/np.sqrt(g*H))
print 'time step: %f'%dt

if eec:
	output_dir = 'output_mimsem_eec'
	we = WaveEqn_EEC(topo,quad,topo_q,lx,g,H,dt)
else:
	output_dir = 'output_mimsem'
	we = WaveEqn(topo,quad,topo_q,lx,g,H)

Mxto0 = Xto0(topo,quad).M
ui = Mxto0*ux
Mxto1 = Xto1(topo,quad).M
hi = Mxto1*hx

uo = np.zeros(n*nx,dtype=np.float64)
ho = np.zeros(n*nx,dtype=np.float64)
uo[:] = ui[:]
ho[:] = hi[:]

uc = np.zeros((21,m*nx),dtype=np.float64)
hc = np.zeros((21,m*nx),dtype=np.float64)

i_dump = 0
plot(hi,ui,x,topo,topo_q,Njxi,Ejxi,i_dump,ho,uo,hc,uc,output_dir)

en0 = energy(hi,ui,g,H,topo,quad,Njxi,Ejxi,det)

for step in np.arange(nsteps) + 1:
	if eec:
		hf,uf = we.solve(hi,ui)
	else:
		hf,uf = we.solveRK2(hi,ui,dt)

	hi[:] = hf[:]
	ui[:] = uf[:]

	if (step%(nsteps/20)==0):
		i_dump = i_dump + 1
		print '\tdumping output for time step %.4d'%step
		plot(hi,ui,x,topo,topo_q,Njxi,Ejxi,i_dump,ho,uo,hc,uc,output_dir)

		en = energy(hi,ui,g,H,topo,quad,Njxi,Ejxi,det)
		print '\t\t' + str(en0) + '\t' + str(en) + '\t' + str(1.0 - en/en0) 

tt = time*np.linspace(0.0,1.0,i_dump+1,endpoint=True)

levs = np.linspace(-1.6,+1.6,101,endpoint=True)
plt.contourf(x,tt,uc,levs)
plt.colorbar()
plt.savefig(output_dir+'/wave_mim_uc.png')
plt.clf()

levs = np.linspace(-0.2,+1.2,101,endpoint=True)
plt.contourf(x,tt,hc,levs)
plt.colorbar()
plt.savefig(output_dir+'/wave_mim_hc.png')
plt.clf()
