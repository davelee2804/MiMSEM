import numpy as np
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from Mats1D import *
from Assembly import *

# Map spatial data to 1 forms
class Xto1:
	def __init__(self,topo,quad,dX):
		M1 = Pmat(topo,quad,dX).M
		M1inv = la.inv(M1)
		PtQ = PtQmat(topo,quad,dX).M
		self.M = M1inv*PtQ

# Map spatial data to 0 forms
class Xto0:
	def __init__(self,topo,quad,dX):
		M0 = Umat(topo,quad,dX).M
		M0inv = la.inv(M0)
		UtQ = UtQmat(topo,quad,dX).M
		self.M = M0inv*UtQ

def plot_h(h,x,topo,topo_q,dX):
	he = np.zeros(topo.n,dtype=np.float64)
	hx = np.zeros(topo_q.n*topo.nx,dtype=np.float64)

	E = LagrangeEdge(topo.n,topo_q.n).M_ij_c

	for ex in np.arange(topo.nx):
		inds1 = topo.localToGlobal1(ex)
		for ii in np.arange(topo.n):
			he[ii] = h[inds1[ii]]

		inds0q = topo_q.localToGlobal0(ex)
		for ii in np.arange(topo_q.n+1):
			if ii == 0 or ii == topo_q.n:
				a = 0.5
			else:
				a = 1.0

			kk = inds0q[ii]
			for jj in np.arange(topo.n):
				hx[kk] = hx[kk] + a*E[ii,jj]*he[jj]*(2.0/dX[ex])

	return hx

def plot_u(u,x,topo,topo_q,dX):
	ue = np.zeros(topo.n+1,dtype=np.float64)
	ux = np.zeros(topo_q.n*topo.nx,dtype=np.float64)

	N = LagrangeNode(topo.n,topo_q.n).M_ij_c

	for ex in np.arange(topo.nx):
		inds0 = topo.localToGlobal0(ex)
		for ii in np.arange(topo.n+1):
			ue[ii] = u[inds0[ii]]

		inds0q = topo_q.localToGlobal0(ex)
		for ii in np.arange(topo_q.n+1):
			if ii == 0 or ii == topo_q.n:
				a = 0.5
			else:
				a = 1.0

			kk = inds0q[ii]
			for jj in np.arange(topo.n+1):
				ux[kk] = ux[kk] + a*N[ii,jj]*ue[jj]

	return ux

