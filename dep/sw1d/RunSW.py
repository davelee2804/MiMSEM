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
from SWEqn_Constrained import *

ne = 16
Lx = 1.0
dX = (Lx/ne)*np.ones(ne)
#dt = 0.1*(Lx/ne)
dt = 1.0*(Lx/ne)
grav = 1.0

N = 3 # polynomial order
M = 5 # quadrature order
topo = Topo(ne,N)
topo_q = Topo(ne,M)
quad = GaussLobatto(M)

X = np.zeros((M*ne))
for ii in np.arange(ne):
	X[ii*M:(ii+1)*M] = np.sum(dX[:ii]) + dX[ii]*0.5*(quad.x[:M]+1.0)

sw = SWEqn(topo,quad,X,dX,dt,grav)

xto1 = Xto1(topo,quad,dX).M
xto0 = Xto0(topo,quad,dX).M

hx = 1.0 + 0.01*np.sin(2.0*np.pi*X)

u1 = np.zeros(N*ne)
h1 = xto1*hx

u2 = np.zeros(N*ne)
h2 = xto1*hx

u3 = np.zeros(N*ne)
h3 = xto1*hx

e0 = sw.energy(u1,h1)

n_steps=100
e1 = np.zeros(n_steps+1)
e1[0] = 0.0
e2 = np.zeros(n_steps+1)
e2[0] = 0.0

dump=0
for step in np.arange(n_steps)+1:
	print('doing step: ' + str(step))
	sw.solve(u1,h1)
	en = sw.energy(u1,h1)
	e1[step] = (e0-en)/e0
	sw.solve_c(u2,h2)
	en = sw.energy(u2,h2)
	e2[step] = (e0-en)/e0
	print('energy conservation: ' + str(e1[step]) + '\t' + str(e2[step]))
	if step%20==0:
		ux = plot_u(u1,X,topo,topo_q,dX)
		plt.plot(X,ux)
		ux = plot_u(u2,X,topo,topo_q,dX)
		plt.plot(X,ux)
		plt.savefig('sw_u_%.4u.png'%dump)
		plt.clf()
		hx = plot_h(h1,X,topo,topo_q,dX)
		plt.plot(X,hx)
		hx = plot_h(h2,X,topo,topo_q,dX)
		plt.plot(X,hx)
		plt.ylim([0.988,1.012])
		plt.savefig('sw_h_%.4u.png'%dump)
		plt.clf()
		dump = dump + 1

plt.semilogy(dt*np.arange(n_steps+1),np.abs(e1))
plt.semilogy(dt*np.arange(n_steps+1),np.abs(e2))
plt.xlabel('time')
plt.ylabel('energy conservation error')
plt.savefig('energy_conservation_newton.pdf')
plt.show()
