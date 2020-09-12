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

def velx_func(x,u):
	u[:] = 0.0

def pres_func(x,h):
	#fac = 20.0
	fac = 200.0
	for ii in np.arange(x.shape[0]):
		if x[ii] < 0.5:
			h[ii] = -0.5 + 0.5*np.tanh(fac*(x[ii] - 0.4))
		else:
			h[ii] = -0.5 + 0.5*np.tanh(fac*(0.6 - x[ii]))

def GenMesh(ne, dXe):
	dX = np.zeros(ne)
	dX[0] = 1.0
	for ii in np.arange(ne-1):
		if ii < ne/2:
			dX[ii+1] = dX[ii]*dXe
		else:
			dX[ii+1] = dX[ii]/dXe

	X = np.zeros(ne+1)
	for ii in np.arange(ne):
		X[ii+1] = X[ii] + dX[ii]

	X = X/X[ne]

	dX2 = np.zeros(ne)
	for ii in np.arange(ne):
		dX2[ii] = X[ii+1] - X[ii]

	dE = np.zeros(ne)
	for ii in np.arange(ne):
		ip1 = (ii+1)%ne
		dE[ii] = dX2[ip1]/dX2[ii]

	#print dE

	return X, dX2

def plot_2(h,u,v,x,topo,topo_q,N,E,step,ho,uo,hc,uc,dX):
	ve = np.zeros(topo.n,dtype=np.float64)
	ue = np.zeros(topo.n,dtype=np.float64)
	he = np.zeros(topo.n,dtype=np.float64)
	uoe = np.zeros(topo.n,dtype=np.float64)
	hoe = np.zeros(topo.n,dtype=np.float64)

	vx = np.zeros(topo_q.n*topo.nx,dtype=np.float64)
	ux = np.zeros(topo_q.n*topo.nx,dtype=np.float64)
	hx = np.zeros(topo_q.n*topo.nx,dtype=np.float64)
	uox = np.zeros(topo_q.n*topo.nx,dtype=np.float64)
	hox = np.zeros(topo_q.n*topo.nx,dtype=np.float64)

	E = LagrangeEdge(topo.n,topo_q.n).M_ij_c

	for ex in np.arange(topo.nx):
		inds0 = topo.localToGlobal0(ex)

		inds1 = topo.localToGlobal1(ex)
		for ii in np.arange(topo.n):
			ve[ii] = v[inds1[ii]]
			ue[ii] = u[inds1[ii]]
			uoe[ii] = uo[inds1[ii]]
			he[ii] = h[inds1[ii]]
			hoe[ii] = ho[inds1[ii]]

		inds0q = topo_q.localToGlobal0(ex)
		for ii in np.arange(topo_q.n+1):
			if ii == 0 or ii == topo_q.n:
				a = 0.5
			else:
				a = 1.0
			kk = inds0q[ii]

			for jj in np.arange(topo.n):
				vx[kk] = vx[kk] + a*E[ii,jj]*ve[jj]*(2.0/dX[ex])
				ux[kk] = ux[kk] + a*E[ii,jj]*ue[jj]*(2.0/dX[ex])
				uox[kk] = uox[kk] + a*E[ii,jj]*uoe[jj]*(2.0/dX[ex])
				hx[kk] = hx[kk] + a*E[ii,jj]*he[jj]*(2.0/dX[ex])
				hox[kk] = hox[kk] + a*E[ii,jj]*hoe[jj]*(2.0/dX[ex])

	uc[step,:] = ux[:]
	hc[step,:] = hx[:]

	#plt.plot(x,uox,'ro')
	#plt.plot(x,hox,'go')

	plt.plot(x,hx,'g-')
	plt.plot(x,vx,'b-')
	plt.plot(x,ux,'r-.')
	plt.legend([r'$A$',r'$A_{PG;\Delta t}$',r'$-A_{PG;-\Delta t}^{\top}$'])
	plt.ylim([-0.2,+1.2])
	plt.xlabel('$x$')
	plt.ylabel('$q$')
	plt.savefig('wave_mim_%.4d'%step + '.pdf')
	if step==20:
		plt.show()
	plt.clf()
