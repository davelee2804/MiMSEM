#!/usr/bin/env python3

import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

from Basis import *
from Topo import *
from Mats1D import *
from Assembly import *
from BoundaryMat import *
from Proj import *
from AdvEqn import *
from Plotting import *

ne = 20
dXe = 1.0#1.25

X,dX=GenMesh(ne,dXe)

# test that the edge functions are a partition of unity (so that boundary terms conserve mass)
N = 3 # polynomial order
M = 3 # quadrature order
topo = Topo(ne,N)
topo_q = Topo(ne,M)
quad = GaussLobatto(M)
edge_ij = M1_j_x_i(topo.n,quad.n).A
quad_ii = Wii(quad.n).A
QE = mult(quad_ii,edge_ij)

one = np.ones(QE.shape[0])
num = np.zeros(QE.shape[1])
for ii in np.arange(QE.shape[0]):
	for jj in np.arange(QE.shape[1]):
		num[jj] = num[jj] + one[ii]*QE[ii][jj]

time = 5.0
nsteps = 1000
dt = time/nsteps

x = np.zeros(ne*M)
for ii in np.arange(ne):
	x[ii*M:ii*M+M] = X[ii] + dX[ii]*0.5*(quad.x[:M]+1)

ux = np.ones(len(x))
hx = np.zeros(len(x))
pres_func(x,hx)
hx = hx + 1.0

Mxto0 = Xto0(topo,quad,dX).M
ui = Mxto0*ux
ui[:] = 0.4
Mxto1 = Xto1(topo,quad,dX).M
hi = Mxto1*hx

ad = AdvEqn(topo,quad,dX,dt,ui)

uo = np.zeros(N*ne,dtype=np.float64)
ho = np.zeros(N*ne,dtype=np.float64)
uo[:] = ui[:]
ho[:] = hi[:]
ht = np.zeros(N*ne,dtype=np.float64)
ht[:] = hi[:]
h2 = np.zeros(N*ne,dtype=np.float64)
h2[:] = hi[:]

uc = np.zeros((21,M*ne),dtype=np.float64)
hc = np.zeros((21,M*ne),dtype=np.float64)

i_dump = 0
Njxi = M0_j_x_i(topo.n,quad.n).A
Ejxi = M1_j_x_i(topo.n,quad.n).A

# eigenvalues for the implicit operator
fig,ax1=plt.subplots()

w1,v = scipy.sparse.linalg.eigs(ad.St,k=len(hi)-2)
wi = w1.imag
inds1 = np.argsort(wi)[::-1]
wii1 = wi[inds1]
plt.plot(w1[inds1].real,wii1,'g.')

w2,v = scipy.sparse.linalg.eigs(ad.Q_up_2,k=len(hi)-2)
wi = w2.imag
inds2 = np.argsort(wi)[::-1]
wii2 = wi[inds2]
plt.plot(w2[inds2].real,wii2,'b.')

w3,v = scipy.sparse.linalg.eigs(ad.Q_up,k=len(hi)-2)
wi = w3.imag
inds3 = np.argsort(wi)[::-1]
wii3 = wi[inds3]
plt.plot(w3[inds3].real,wii3,'r+')

xx=np.linspace(0.0,1.0,8193)
plt.plot(xx,np.sqrt(1.0-xx*xx),c='k')
plt.plot(xx,-np.sqrt(1.0-xx*xx),c='k')
plt.xlim([0,2])
plt.ylim([-1,+1])
plt.title('trapazoidal advection operator eigenvalues')
plt.legend([r'$A$',r'$A_{PG;\Delta t}$',r'$-A_{PG;-\Delta t}^{\top}$'])
plt.xlabel('real component, $\omega^r$')
plt.ylabel('imaginary component, $\omega^i$')

ax2 = plt.axes([0,0,1,1])
ip = InsetPosition(ax1, [0.6,0.1,0.3,0.6])
ax2.set_axes_locator(ip)
mark_inset(ax1,ax2,loc1=2,loc2=3,fc="none",ec='0.5')
ax2.plot(xx,np.sqrt(1.0-xx*xx),c='k')
ax2.plot(xx,-np.sqrt(1.0-xx*xx),c='k')
ax2.plot(w1[inds1].real,wii1,'g.')
ax2.plot(w2[inds2].real,wii2,'b.')
ax2.plot(w3[inds3].real,wii3,'r+')
ax2.set_xlim(0.90,1.01)
ax2.set_ylim(-0.40,+0.40)

plt.savefig('stab_regions.pdf')
plt.show()

kc,wc = ad.disp_rel(ad.A_cen,False)
ku,wu = ad.disp_rel(ad.A_upw,False)
kd,wd = ad.disp_rel(ad.A_up2,False)
plt.plot((1.0*kc)/(0.5*ne*N),ui[0]*np.abs(wc)/(0.5*ne*N),'go')
plt.plot((1.0*kd)/(0.5*ne*N),ui[0]*np.abs(wd)/(0.5*ne*N),'bo')
plt.plot((1.0*ku)/(0.5*ne*N),ui[0]*np.abs(wu)/(0.5*ne*N),'r+')
plt.plot((1.0*kc)/(0.5*ne*N),(1.0*kc)/(0.5*ne*N),c='k')
plt.title('advection operator eigenvalues (imaginary component)')
plt.legend([r'$A$',r'$A_{PG;\Delta t}$',r'$-A_{PG;-\Delta t}^{\top}$'])
plt.ylabel('angular frequency, $\omega$')
plt.xlabel('wavenumber, $k$')
plt.savefig('eig_vals_imag.png')
plt.show()
kc,wc = ad.disp_rel(ad.A_cen,True)
ku,wu = ad.disp_rel(ad.A_upw,True)
kd,wd = ad.disp_rel(ad.A_up2,True)
plt.plot((1.0*kc)/(0.5*ne*N),np.abs(wc)/(0.5*ne*N*2.0*np.pi),'go')
plt.plot((1.0*kd)/(0.5*ne*N),np.abs(wd)/(0.5*ne*N*2.0*np.pi),'bo')
plt.plot((1.0*ku)/(0.5*ne*N),np.abs(wu)/(0.5*ne*N*2.0*np.pi),'r+')
plt.title('advection operator eigenvalues (real component)')
plt.legend([r'$A$',r'$A_{PG;\Delta t}$',r'$-A_{PG;-\Delta t}^{\top}$'])
plt.ylabel('angular frequency, $\omega$')
plt.xlabel('wavenumber, $k$')
plt.savefig('eig_vals_real.png')
plt.show()

nt = 64
A_cen = np.zeros((nt,len(hi)-2))
A_upw = np.zeros((nt,len(hi)-2))
A_dwn = np.zeros((nt,len(hi)-2))
for i in np.arange(nt):
	dt = 0.0005*i
	print('time step: ' + str(dt))
	ad = AdvEqn(topo,quad,dX,dt,ui)

	kc,wc = ad.disp_rel(ad.A_cen,False)
	ku,wu = ad.disp_rel(ad.A_upw,False)
	kd,wd = ad.disp_rel(ad.A_up2,False)

	w1,v = scipy.sparse.linalg.eigs(ad.St,k=len(hi)-2)
	wi = w1.imag
	inds1 = np.argsort(wi)[::-1]
	w1s = w1[inds1]
	for j in np.arange(len(hi)-2):
		A_cen[i,j] = np.sqrt(w1s[j].real*w1s[j].real + w1s[j].imag*w1s[j].imag)

	w1,v = scipy.sparse.linalg.eigs(ad.Q_up_2,k=len(hi)-2)
	wi = w1.imag
	inds1 = np.argsort(wi)[::-1]
	w1s = w1[inds1]
	for j in np.arange(len(hi)-2):
		A_upw[i,j] = np.sqrt(w1s[j].real*w1s[j].real + w1s[j].imag*w1s[j].imag)

	w1,v = scipy.sparse.linalg.eigs(ad.Q_up,k=len(hi)-2)
	wi = w1.imag
	inds1 = np.argsort(wi)[::-1]
	w1s = w1[inds1]
	for j in np.arange(len(hi)-2):
		A_dwn[i,j] = np.sqrt(w1s[j].real*w1s[j].real + w1s[j].imag*w1s[j].imag)

plt.contourf(A_cen.transpose(),100)
plt.colorbar()
plt.show()
plt.contourf(A_upw.transpose(),100)
plt.colorbar()
plt.show()
plt.contourf(A_dwn.transpose(),100)
plt.colorbar()
plt.show()
