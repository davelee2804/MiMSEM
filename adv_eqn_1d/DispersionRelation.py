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
from Plotting import *

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
	g = 1.0
	H = 1.0
	k = 2.0*np.pi
	dXe = 1.0
	X,dX=GenMesh(nx,dXe)
	
	x = np.zeros(n*nx)
	for ii in np.arange(nx):
		x[ii*n:ii*n+n] = X[ii] + dX[ii]*0.5*(quad.x[:n]+1)

	omega = k*np.sqrt(g*H)

	we = WaveEqn(topo,quad,g,H,dX,1.0)

	A = np.zeros((nx*n,nx*n),dtype=np.float64)
	kk = k*np.arange(nx*n)/lx
	np.fill_diagonal(A,1.0)

	B = we.M

	vals,vecs = la.eigs(B,M=A,k=nx*n-2)
	vr = vals.real
	inds = np.argsort(vr)[::-1]
	vr2 = vr[inds]
	vecr = vecs.real
	vecr2 = vecr[:,inds]

	return vr2, vecr2, x

def fourier_decomp(vals,vecs,x):
	nx = x.shape[0]
	nv = vecs.shape[0]
	A = np.zeros((nx,nx),dtype=np.complex128)

	xx = 2.0*np.pi*x
	kk = 1.0*(np.arange(nx) - nx/2 + 1)
	x2 = (2.0*np.pi/nx)*np.arange(nx)

	for ii in np.arange(nx):
		for jj in np.arange(nx):
			A[ii,jj] = np.exp(1.0j*kk[jj]*xx[ii])

	Ainv = np.linalg.inv(A)

	ki = np.zeros((nx-2,8),dtype=np.int32)

	for ii in np.arange(nx-2):
		vf = np.dot(Ainv,vecs[:,ii])
		inds = np.argsort(np.abs(vf.real))[::-1]
		ki[ii] = np.abs(kk[inds[0]])

	return ki

vals2, vecs2, x2 = disp_rel(150,2,2)
vals3, vecs3, x3 = disp_rel(100,3,3)
vals4, vecs4, x4 = disp_rel( 75,4,4)
vals5, vecs5, x5 = disp_rel( 60,5,5)
vals6, vecs6, x6 = disp_rel( 50,6,6)

g = 1.0
H = 1.0
lx = 1.0
k = 2.0*np.pi
kk = k*np.arange(300)/lx

# plot the lowest eigenfunctions
xx = x6
vv = vecs6

plt.figure()
plt.subplot(2, 3, 1)
plt.plot(xx,vv[:,0].real)
plt.xticks([])
plt.yticks([])
plt.subplot(2, 3, 2)
plt.plot(xx,vv[:,2].real)
plt.xticks([])
plt.yticks([])
plt.subplot(2, 3, 3)
plt.plot(xx,vv[:,4].real)
plt.xticks([])
plt.yticks([])
plt.subplot(2, 3, 4)
plt.plot(xx,vv[:,6].real)
plt.xticks([])
plt.yticks([])
plt.subplot(2, 3, 5)
plt.plot(xx,vv[:,8].real)
plt.xticks([])
plt.yticks([])
plt.subplot(2, 3, 6)
plt.plot(xx,vv[:,10].real)
plt.xticks([])
plt.yticks([])
plt.savefig('eigenfuncs_6_mixed.png')

plt.show()

# re-order the eigenvalues based on fourier decomposition
k2 = fourier_decomp(vals2,vecs2,x2)
k3 = fourier_decomp(vals3,vecs3,x3)
k4 = fourier_decomp(vals4,vecs4,x4)
k5 = fourier_decomp(vals5,vecs5,x5)
k6 = fourier_decomp(vals6,vecs6,x6)

plt.subplot(2, 3, 1)
plt.plot((kk[:(300)/2])/2.0/np.pi,np.sqrt(g*H)*kk[:(300)/2],c='k')
plt.plot(k2,np.sqrt(np.abs(vals2)),'.')
plt.plot(k3,np.sqrt(np.abs(vals3)),'.')
plt.plot(k4,np.sqrt(np.abs(vals4)),'.')
plt.plot(k5,np.sqrt(np.abs(vals5)),'.')
plt.plot(k6,np.sqrt(np.abs(vals6)),'.')
plt.ylim([0.0,1400.0])
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 2)
plt.title('$p=2$')
plt.plot((kk[:(300)/2])/2.0/np.pi,np.sqrt(g*H)*kk[:(300)/2],c='k')
plt.plot(k2,np.sqrt(np.abs(vals2)),'.')
plt.ylim([0.0,1400.0])
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 3)
plt.title('$p=3$')
plt.plot((kk[:(300)/2])/2.0/np.pi,np.sqrt(g*H)*kk[:(300)/2],c='k')
plt.plot(k3,np.sqrt(np.abs(vals3)),'.')
plt.ylim([0.0,1400.0])
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 4)
plt.title('$p=4$')
plt.plot((kk[:(300)/2])/2.0/np.pi,np.sqrt(g*H)*kk[:(300)/2],c='k')
plt.plot(k4,np.sqrt(np.abs(vals4)),'.')
plt.ylim([0.0,1400.0])
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 5)
plt.title('$p=5$')
plt.plot((kk[:(300)/2])/2.0/np.pi,np.sqrt(g*H)*kk[:(300)/2],c='k')
plt.plot(k5,np.sqrt(np.abs(vals5)),'.')
plt.ylim([0.0,1400.0])
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 6)
plt.title('$p=6$')
plt.plot((kk[:(300)/2])/2.0/np.pi,np.sqrt(g*H)*kk[:(300)/2],c='k')
plt.plot(k6,np.sqrt(np.abs(vals6)),'.')
plt.ylim([0.0,1400.0])
plt.xticks([])
plt.yticks([])

plt.savefig('dispersion_relations_mixed.png')

plt.show()
