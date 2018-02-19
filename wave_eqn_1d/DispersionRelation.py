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
from WaveEqn_StdSEM import *

# dispersion relation: w = k(gH)**0.5; c_p = (gH)**0.5

def disp_rel(nx,n,m,mixed):
	np1 = n+1
	mp1 = m+1
	nxn = nx*n
	nxm = nx*m

	topo = Topo(nx,n)
	if mixed:
		quad = GaussLegendre(m)
	else:
		quad = GaussLobatto(m)
	topo_q = Topo(nx,m)

	lx = 1.0
	dx = lx/nx
	x = np.zeros(n*nx)
	for i in np.arange(nx):
	    x[i*n:(i+1)*n] = i*dx + (quad.x[:n] + 1.0)*0.5*dx

	det = 0.5*lx/topo.nx

	g = 1.0
	H = 1.0
	k = 2.0*np.pi
	omega = k*np.sqrt(g*H)

	if mixed:
		we = WaveEqn(topo,quad,topo_q,lx,g,H)
	else:
		we = WaveEqn_StdSEM(topo,quad,topo_q,lx,g,H)

	A = np.zeros((nx*n,nx*n),dtype=np.float64)
	kk = k*np.arange(nx*n)/lx
	np.fill_diagonal(A,1.0)

	if mixed:
		B = g*H*we.A*we.B
	else:
		B = g*H*we.A*we.A

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
	#vf = np.dot(Ainv,vecs[:,1])

	#inds = np.argsort(np.abs(vf.real))[::-1]
	#print '%u'%kk[inds[0]] + '\t%2f'%vf[inds[0]].real +  '\t%u'%kk[inds[1]] + '\t%2f'%vf[inds[1]].real + \
	#    '\t%u'%kk[inds[2]] + '\t%2f'%vf[inds[2]].real +  '\t%u'%kk[inds[3]] + '\t%2f'%vf[inds[3]].real + \
	#    '\t%u'%kk[inds[4]] + '\t%2f'%vf[inds[4]].real +  '\t%u'%kk[inds[5]] + '\t%2f'%vf[inds[5]].real

	#vx2 = np.zeros(nx,dtype=np.float64)
	#for ii in np.arange(nx):
	#	jj = inds[ii]
	#	vx2[:] += vf[jj].real*np.cos(kk[jj]*x2[:]) - vf[jj].imag*np.sin(kk[jj]*x2[:])

	#plt.plot(x,vecs[:,1])
	#plt.plot(x2/2.0/np.pi,vx2)
	#plt.show()

	ki = np.zeros((nx-2,8),dtype=np.int32)

	for ii in np.arange(nx-2):
		vf = np.dot(Ainv,vecs[:,ii])
		inds = np.argsort(np.abs(vf.real))[::-1]
		ki[ii] = np.abs(kk[inds[0]])

	return ki

inexact = True
mixed = True

if inexact:
	vals2, vecs2, x2 = disp_rel(150,2,2,mixed)
	vals3, vecs3, x3 = disp_rel(100,3,3,mixed)
	vals4, vecs4, x4 = disp_rel( 75,4,4,mixed)
	vals5, vecs5, x5 = disp_rel( 60,5,5,mixed)
	vals6, vecs6, x6 = disp_rel( 50,6,6,mixed)
else:
	vals2, vecs2, x2 = disp_rel(150,2,2+2,mixed)
	vals3, vecs3, x3 = disp_rel(100,3,3+2,mixed)
	vals4, vecs4, x4 = disp_rel( 75,4,4+2,mixed)

g = 1.0
H = 1.0
lx = 1.0
k = 2.0*np.pi
kk = k*np.arange(300)/lx

plt.plot((kk[:(300)/2])/2.0/np.pi,np.sqrt(g*H)*kk[:(300)/2],c='k')
plt.plot((2.0*np.pi+0.5*kk[:-2])/2.0/np.pi,np.sqrt(np.abs(vals2)))
plt.plot((2.0*np.pi+0.5*kk[:-2])/2.0/np.pi,np.sqrt(np.abs(vals3)))
plt.plot((2.0*np.pi+0.5*kk[:-2])/2.0/np.pi,np.sqrt(np.abs(vals4)))
if inexact:
	plt.plot((2.0*np.pi+0.5*kk[:-2])/2.0/np.pi,np.sqrt(np.abs(vals5)))
	plt.plot((2.0*np.pi+0.5*kk[:-2])/2.0/np.pi,np.sqrt(np.abs(vals6)))
plt.legend(['analytic','p=2','p=3','p=4','p=5','p=6'],loc='upper left')
plt.xlabel('$k$')
plt.ylabel('$\omega$')
plt.ylim([0.0,1600])
if inexact:
	plt.savefig('dispersion_relation_inexact_quadrature.png')
else:
	plt.savefig('dispersion_relation_exact_quadrature.png')
plt.show()

xx = x4
vv = vecs4

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
if mixed:
	plt.savefig('eigenfuncs_4_mixed.png')
else:
	plt.savefig('eigenfuncs_4_agrid.png')

plt.show()

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

if mixed:
	plt.savefig('dispersion_relations_mixed.png')
else:
	plt.savefig('dispersion_relations_agrid.png')

plt.show()
