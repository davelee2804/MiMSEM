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

# dispersion relation: w = k(gH)**0.5; c_p = (gH)**0.5

def disp_rel(nx,n,m):
	np1 = n+1
	mp1 = m+1
	nxn = nx*n
	nxm = nx*m

	topo = Topo(nx,n)
	quad = GaussLobatto(m)
	topo_q = Topo(nx,m)

	lx = 1.0
	dx = lx/nx
	x = np.zeros(n*nx)
	for i in np.arange(nx):
	    x[i*n:(i+1)*n] = i*dx + (quad.x[:n] + 1.0)*0.5*dx

	det = 0.5*lx/topo.nx

	g = 10.0
	H = 1.6
	k = 2.0*np.pi
	omega = k*np.sqrt(g*H)

	we = WaveEqn(topo,quad,topo_q,lx,g,H)

	A = np.zeros((nx*n,nx*n),dtype=np.float64)
	kk = k*np.arange(nx*n)/lx
	np.fill_diagonal(A,1.0)

	B = g*H*we.A*we.B

	vals,vecs = la.eigs(B,M=A,k=nx*n-2)
	vr = vals.real
	inds = np.argsort(vr)[::-1]
	vr2 = vr[inds]
	vecr = vecs.real
	vecr2 = vecr[:,inds]

	return vr2, vecr2

vals2, vecs2 = disp_rel(150,2,2)
vals3, vecs3 = disp_rel(100,3,3)
vals4, vecs4 = disp_rel( 75,4,4)
vals5, vecs5 = disp_rel( 60,5,5)
vals6, vecs6 = disp_rel( 50,6,6)

#vals2, vecs2 = disp_rel(150,2,2+2)
#vals3, vecs3 = disp_rel(100,3,3+2)
#vals4, vecs4 = disp_rel( 75,4,4+2)

g = 10.0
H = 1.6
lx = 1.0
k = 2.0*np.pi
kk = k*np.arange(300)/lx

plt.plot((kk[:(300)/2])/2.0/np.pi,np.sqrt(g*H)*kk[:(300)/2],c='k')
plt.plot((2.0*np.pi+0.5*kk[:-2])/2.0/np.pi,np.sqrt(np.abs(vals2)))
plt.plot((2.0*np.pi+0.5*kk[:-2])/2.0/np.pi,np.sqrt(np.abs(vals3)))
plt.plot((2.0*np.pi+0.5*kk[:-2])/2.0/np.pi,np.sqrt(np.abs(vals4)))
plt.plot((2.0*np.pi+0.5*kk[:-2])/2.0/np.pi,np.sqrt(np.abs(vals5)))
plt.plot((2.0*np.pi+0.5*kk[:-2])/2.0/np.pi,np.sqrt(np.abs(vals6)))
plt.legend(['analytic','p=2','p=3','p=4','p=5','p=6'],loc='upper left')
plt.xlabel('wavenumber')
plt.ylabel('phase speed')
#plt.savefig('dispersion_relation_inexact_quadrature.png')
plt.show()

#plt.plot(x,vecr2[:,0].real)
#plt.plot(x,vecr2[:,1].real)
#plt.plot(x,vecr2[:,2].real)
#plt.plot(x,vecr2[:,3].real)
#plt.plot(x,vecr2[:,4].real)
#plt.show()
