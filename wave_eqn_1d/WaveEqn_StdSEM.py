import numpy as np
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from Mats1D import *
from Assembly import *
from BoundaryMat import *
from Proj import *

class WaveEqn_StdSEM:
	def __init__(self,topo,quad,topo_q,lx,g,H):
		self.topo = topo
		self.quad = quad
		self.topo_q = topo_q # topology for the quadrature points as 0 forms
		self.g = g
		self.H = H
		det = 0.5*lx/topo.nx
		self.detInv = 1.0/det

		# 0 form matrix inverse
		self.M0 = Umat(topo,quad).M
		self.M0inv = la.inv(self.M0)

		# Gradient operator
		self.G = GradSEM(topo,quad).M

		# Divergence operator
		self.D = DivSEM(topo,quad).M

		# 0 form to 1 from gradient operator
		self.A = +1.0*self.detInv*self.M0inv*self.D

		# 1 form to 0 form gradient operator
		self.B = -1.0*self.detInv*self.M0inv*self.G

	def prognose_h(self,hi,ud,dt):
		hf = hi - dt*self.H*self.A*ud
		return hf

	def prognose_u(self,hi,ui,hd,ud,dt):
		#uf = ui - dt*self.g*self.B*hd
		uf = ui - dt*self.g*self.A*hd
		return uf

	def solveEuler(self,hi,ui,hd,ud,dt):
		hf = self.prognose_h(hi,ud,dt)
		uf = self.prognose_u(hi,ui,hd,ud,dt)

		return hf,uf

	def solveRK2(self,hi,ui,dt):
		hh,uh = self.solveEuler(hi,ui,hi,ui,0.5*dt)
		hf,uf = self.solveEuler(hi,ui,hh,uh,dt)

		return hf,uf
