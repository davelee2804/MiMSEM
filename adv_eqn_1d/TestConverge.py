#!/usr/bin/env python

import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt

from Basis import *
from Topo import *
from Mats1D import *
from Assembly import *
from BoundaryMat import *
from Proj import *
from AdvEqn import *
from Plotting import *

scale = int(sys.argv[2])
ne = 4*scale
dXe = 1.0

X,dX=GenMesh(ne,dXe)

# test that the edge functions are a partition of unity (so that boundary terms conserve mass)
N = int(sys.argv[1]) # polynomial order
M = int(sys.argv[1]) # quadrature order
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

time = 2.5
#nsteps = 20*scale
nsteps = 40*scale
dt = time/nsteps

x = np.zeros(ne*M)
for ii in np.arange(ne):
	x[ii*M:ii*M+M] = X[ii] + dX[ii]*0.5*(quad.x[:M]+1)

ux = np.ones(len(x))
hx = -0.5*(np.cos(2.0*np.pi*x)-1.0)

Mxto0 = Xto0(topo,quad,dX).M
ui = Mxto0*ux
ui[:] = 0.4
Mxto1 = Xto1(topo,quad,dX).M
hi = Mxto1*hx

uo = np.zeros(N*ne,dtype=np.float64)
ho = np.zeros(N*ne,dtype=np.float64)
uo[:] = ui[:]
ho[:] = hi[:]
ht = np.zeros(N*ne,dtype=np.float64)
ht[:] = hi[:]
h2 = np.zeros(N*ne,dtype=np.float64)
h2[:] = hi[:]

ui[:] = 0.4 + 0.2*(np.sin(2.0*np.pi*x)+1.0)
F = hx*ui

ad = AdvEqn(topo,quad,dX,dt,ui)

F_a = ad.B*hi
F_1 = ad.B_dep*hi
F_2 = ad.B_up2*hi
plt.plot(x,F_a,c='g')
plt.plot(x,F_1,c='r')
plt.plot(x,F_2,c='b')
plt.plot(x,F,c='k')
plt.show()

F_a = F_a - F
F_1 = F_1 - F
F_2 = F_2 - F

a=ad.M0*F_a
err_1 = np.sqrt(np.dot(a,F_a))
a=ad.M0*F_1
err_2 = np.sqrt(np.dot(a,F_1))
a=ad.M0*F_2
err_3 = np.sqrt(np.dot(a,F_2))
print str(err_1) + '\t' + str(err_2) + '\t' + str(err_3)


#uc = np.zeros((21,M*ne),dtype=np.float64)
#hc = np.zeros((21,M*ne),dtype=np.float64)
#
#i_dump = 0
#Njxi = M0_j_x_i(topo.n,quad.n).A
#Ejxi = M1_j_x_i(topo.n,quad.n).A
#plot_2(hi,ht,h2,x,topo,topo_q,Njxi,Ejxi,i_dump,ho,uo,hc,uc,dX)
#
#mass_a=np.zeros(nsteps+1)
#mass_t=np.zeros(nsteps+1)
#mass_2=np.zeros(nsteps+1)
#
#_one = np.ones(ho.shape)
#
#energy=np.zeros(nsteps+1)
#a=ad.M1*hi
#mass_a[0]=np.dot(_one,a)
#energy[0]=np.dot(a,hi)
#energyt=np.zeros(nsteps+1)
#a=ad.M1*hi
#mass_t[0]=np.dot(_one,a)
#energyt[0]=np.dot(a,hi)
#energy2=np.zeros(nsteps+1)
#a=ad.M1*hi
#mass_2[0]=np.dot(_one,a)
#energy2[0]=np.dot(a,hi)
#
#for step in np.arange(nsteps) + 1:
#	hf = ad.solve_a(hi)
#	hft = ad.solve_a_up(ht)
#	hf2 = ad.solve_2(h2)
#
#	hi[:] = hf[:]
#	ht[:] = hft[:]
#	h2[:] = hf2[:]
#
#	if (step%(nsteps/20)==0):
#		i_dump = i_dump + 1
#		print '\tdumping output for time step %.4d'%step
#		plot_2(hi,ht,h2,x,topo,topo_q,Njxi,Ejxi,i_dump,ho,uo,hc,uc,dX)
#
#	a=ad.M1*hi
#	mass_a[step]=np.dot(_one,a)
#	energy[step]=np.dot(a,hi)
#	a=ad.M1*ht
#	mass_t[step]=np.dot(_one,a)
#	energyt[step]=np.dot(a,ht)
#	a=ad.M1*h2
#	mass_2[step]=np.dot(_one,a)
#	energy2[step]=np.dot(a,h2)
#
#node = LagrangeNode(N,M)
#edge = LagrangeEdge(N,M)
#QP = P_interp(topo,quad,dX,node,edge).M
#hf = QP*hf - QP*ho
#ht = QP*ht - QP*ho
#h2 = QP*h2 - QP*ho
#a=ad.M0*hf
#err_1 = np.sqrt(np.dot(a,hf))
#a=ad.M0*ht
#err_2 = np.sqrt(np.dot(a,ht))
#a=ad.M0*h2
#err_3 = np.sqrt(np.dot(a,h2))
#print str(err_1) + '\t' + str(err_2) + '\t' + str(err_3)
#
#plt.plot((mass_a-mass_a[0])/mass_a[0],c='g')
#plt.plot((mass_t-mass_t[0])/mass_t[0],c='r')
#plt.plot((mass_2-mass_2[0])/mass_2[0],c='b')
#plt.title('mass conservation')
#plt.legend(['A','A_extrusion','A_upwind_basis'])
#plt.savefig('mass_conservation.png')
#plt.show()
#plt.plot((energy-energy[0])/energy[0],c='g')
#plt.plot((energyt-energyt[0])/energyt[0],c='r')
#plt.plot((energy2-energy2[0])/energy2[0],c='b')
#plt.title('energy conservation')
#plt.legend(['A','A_extrusion','A_upwind_basis'])
#plt.savefig('energy_conservation.png')
#plt.show()
#
#tt = time*np.linspace(0.0,1.0,i_dump+1,endpoint=True)
#
#levs = np.linspace(-0.2,+1.2,101,endpoint=True)
#plt.contourf(x,tt,uc,100)
#plt.colorbar()
#plt.savefig('adv_mim_hc.png')
#plt.clf()
#
