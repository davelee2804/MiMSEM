import numpy as np
import scipy.sparse.linalg as la
from scipy.sparse import bmat, diags

from Basis import *
from Topo import *
from Mats1D import *
from Assembly import *
from BoundaryMat import *
from Proj import *

# Exact energy conserving form of the linear wave equation, solved
# as a semi-implicit system
class WaveEqn_EEC:
	def __init__(self,topo,quad,topo_q,lx,g,H,dt):
		self.topo = topo
		self.quad = quad
		self.topo_q = topo_q # topology for the quadrature points as 0 forms
		self.g = g
		self.H = H
		self.dt = dt
		det = 0.5*lx/topo.nx
		self.detInv = 1.0/det

		# 1 form matrix inverse
		self.M1 = Pmat(topo,quad).M
		self.M1inv = la.inv(self.M1)

		# 0 form matrix inverse
		self.M0 = Umat(topo,quad).M
		self.M0inv = la.inv(self.M0)

		# 1 form gradient matrix
		self.D10 = BoundaryMat(topo).M
		self.D01 = self.D10.transpose()

		# 0 form to 1 from gradient operator
		self.A = self.detInv*self.D10

		# 1 form to 0 form gradient operator
		self.B = -1.0*self.detInv*self.M0inv*self.D01*self.M1

		# Identity matrix
		self.I = diags([1.0],[0],shape=(topo.nx*topo.n,topo.nx*topo.n))

		# block system
		self.M = bmat([[self.I,0.5*dt*H*self.A],[0.5*dt*g*self.B,self.I]],format='csc',dtype=np.float64)
		self.Minv = la.inv(self.M)

	def solve(self,hi,ui):
		rhs = np.zeros(2*self.topo.nx*self.topo.n,dtype=np.float64)
		rhs[:self.topo.nx*self.topo.n] = hi - 0.5*self.dt*self.H*self.A*ui
		rhs[self.topo.nx*self.topo.n:] = ui - 0.5*self.dt*self.g*self.B*hi

		ans = self.Minv*rhs
		hf = ans[:self.topo.nx*self.topo.n]
		uf = ans[self.topo.nx*self.topo.n:]

		return hf,uf
