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
#ne = 6
dXe = 1.0#1.25

X,dX=GenMesh(ne,dXe)
#print X
#print dX

#plt.plot(X,np.zeros(X.shape[0]),'o')
#plt.plot(np.linspace(0.0,1.0,ne+1,endpoint=True),np.zeros(X.shape[0]),'+',c='r')
#plt.show()

# test that the edge functions are a partition of unity (so that boundary terms
# conserve mass)
N = 5 # polynomial order
M = 5 # quadrature order
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

#print num

time = 5.0
#nsteps = 400
nsteps = 1000
dt = time/nsteps
nsteps = nsteps*10

x = np.zeros(ne*M)
for ii in np.arange(ne):
	x[ii*M:ii*M+M] = X[ii] + dX[ii]*0.5*(quad.x[:M]+1)

ux = np.ones(len(x))
hx = np.zeros(len(x))
pres_func(x,hx)
hx = hx + 1.0

Mxto0 = Xto0(topo,quad,dX).M
ui = Mxto0*ux
#ui[:] = 1.0
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
plot_2(hi,ht,h2,x,topo,topo_q,Njxi,Ejxi,i_dump,ho,uo,hc,uc,dX)

mass_a=np.zeros(nsteps+1)
mass_t=np.zeros(nsteps+1)
mass_2=np.zeros(nsteps+1)

_one = np.ones(ho.shape)

energy=np.zeros(nsteps+1)
a=ad.M1*hi
mass_a[0]=np.dot(_one,a)
energy[0]=np.dot(a,hi)
energyt=np.zeros(nsteps+1)
a=ad.M1*hi
mass_t[0]=np.dot(_one,a)
energyt[0]=np.dot(a,hi)
energy2=np.zeros(nsteps+1)
a=ad.M1*hi
mass_2[0]=np.dot(_one,a)
energy2[0]=np.dot(a,hi)

for step in np.arange(nsteps) + 1:
	hf = ad.solve_a(hi)
	hft = ad.solve_a_up(ht)
	hf2 = ad.solve_2(h2)

	hi[:] = hf[:]
	ht[:] = hft[:]
	h2[:] = hf2[:]

	if (step%(nsteps/20)==0):
		i_dump = i_dump + 1
		print('\tdumping output for time step %.4d'%step)
		plot_2(hi,ht,h2,x,topo,topo_q,Njxi,Ejxi,i_dump,ho,uo,hc,uc,dX)

	a=ad.M1*hi
	mass_a[step]=np.dot(_one,a)
	energy[step]=np.dot(a,hi)
	a=ad.M1*ht
	mass_t[step]=np.dot(_one,a)
	energyt[step]=np.dot(a,ht)
	a=ad.M1*h2
	mass_2[step]=np.dot(_one,a)
	energy2[step]=np.dot(a,h2)

hf = hf - ho
ht = ht - ho
h2 = h2 - ho
a=ad.M1*hf
err_1 = np.sqrt(np.dot(a,hf))
a=ad.M1*ht
err_2 = np.sqrt(np.dot(a,ht))
a=ad.M1*h2
err_3 = np.sqrt(np.dot(a,h2))
print(str(err_1) + '\t' + str(err_2) + '\t' + str(err_3))

plt.plot(dt*np.arange(nsteps+1),(mass_a-mass_a[0])/mass_a[0],c='g')
plt.plot(dt*np.arange(nsteps+1),(mass_2-mass_2[0])/mass_2[0],c='b')
plt.plot(dt*np.arange(nsteps+1),(mass_t-mass_t[0])/mass_t[0],c='r')
plt.title('mass conservation')
plt.legend([r'$A$',r'$A_{PG;\Delta t}$',r'$-A_{PG;-\Delta t}^{\top}$'])
plt.xlabel('time, $t$')
plt.savefig('mass_conservation.pdf')
plt.show()
plt.plot(dt*np.arange(nsteps+1),(energy-energy[0])/energy[0],'g-')
plt.plot(dt*np.arange(nsteps+1),(energy2-energy2[0])/energy2[0],'b-')
plt.plot(dt*np.arange(nsteps+1),(energyt-energyt[0])/energyt[0],'r-')
plt.title('energy conservation')
plt.legend([r'$A$',r'$A_{PG;\Delta t}$',r'$-A_{PG;-\Delta t}^{\top}$'])
plt.xlabel('time, $t$')
plt.savefig('energy_conservation.pdf')
plt.show()

tt = time*np.linspace(0.0,1.0,i_dump+1,endpoint=True)

levs = np.linspace(-0.2,+1.2,101,endpoint=True)
#plt.contourf(x,tt,hc,levs)
plt.contourf(x,tt,uc,100)
plt.colorbar()
plt.savefig('adv_mim_hc.png')
plt.clf()

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

#xx=np.linspace(0.4,1.0,8193)
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
