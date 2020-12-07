#!/usr/bin/env python3

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
F_1 = ad.B_up2*hi
plt.plot(x,F_a,c='g')
plt.plot(x,F_1,c='r')
plt.plot(x,F,c='k')
plt.show()

F_a = F_a - F
F_1 = F_1 - F

a=ad.M0*F_a
err_1 = np.sqrt(np.dot(a,F_a))
a=ad.M0*F_1
err_2 = np.sqrt(np.dot(a,F_1))
print(str(err_1) + '\t' + str(err_2))

node = LagrangeNode(topo.n,quad.n)
edge = LagrangeEdge(topo.n,quad.n)
QP = P_interp_2(topo,quad,dX,node,edge).M

G = ui*np.pi*np.sin(2.0*np.pi*x)
#G = np.pi*np.sin(2.0*np.pi*x)
Gi = Mxto1*G

G_a = ad.C*hi
G_1 = ad.C_up2*hi
G_a = QP*G_a
G_1 = QP*G_1

plt.plot(x,G_a,'o',c='g')
plt.plot(x,G_1,c='r')
plt.plot(x,G,c='k')
plt.show()

G_a = G_a - G
G_1 = G_1 - G
a=ad.M0*G_a
err_1 = np.sqrt(np.dot(a,G_a))
a=ad.M0*G_1
err_2 = np.sqrt(np.dot(a,G_1))
print(str(err_1) + '\t' + str(err_2))
