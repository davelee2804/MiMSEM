import numpy as np
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from Mats2D import *
from Assembly import *
from BoundaryMat import *
from Proj import *

class SWEqn:
	def __init__(self,topo,quad,topo_q,lx,ly,f,g,apvm,hb):
		self.topo = topo
		self.quad = quad
		self.topo_q = topo_q # topology for the quadrature points as 0 forms
		self.g = g
		self.apvm = apvm
		det = 0.5*lx/topo.nx
		self.detInv = 1.0/det
		self.hb = hb

		# 1 form matrix inverse
		M1 = Umat(topo,quad).M
		self.M1inv = la.inv(M1)

		# 2 form matrix inverse
		M2 = Wmat(topo,quad).M
		self.M2inv = la.inv(M2)

		# 0 form matrix inverse
		self.M0 = Pmat(topo,quad).M
		M0inv = la.inv(self.M0)
		D10 = BoundaryMat10(topo).M
		D01 = D10.transpose()
		self.D01M1 = self.detInv*D01*M1

		self.D10 = self.detInv*D10
		self.M0inv = M0inv

		# 2 form gradient matrix
		self.D21 = BoundaryMat(topo).M
		self.D12 = -1.0*self.D21.transpose()
		self.D12M2 = self.detInv*self.D12*M2

		# 0 form coriolis vector
		Mxto0 = Xto0(topo,quad).M
		fx = f*np.ones((topo_q.n*topo_q.n*topo.nx*topo.ny),dtype=np.float64)
		self.f = Mxto0*fx

		self.M0f = self.M0*self.f

		self.Uh = Uhmat(topo,quad)
		self.WU = WtQUmat(topo,quad)
		self.PU = PtQUmat(topo,quad)
		self.Rq = RotationalMat(topo,quad)

		self.q = np.zeros((self.M0f.shape[0]),dtype=np.float64)
		self.hvec = Phvec(topo,quad)

	def diagnose_q(self,h,u):
		w = self.D01M1*u
		#f = self.M0*self.f
		#M0h = Phmat(self.topo,self.quad,h).M
		#M0hinv = la.inv(M0h)
		#q = M0hinv*(w+f)
		hv = self.hvec.assemble(h)
		for ii in np.arange(hv.shape[0]):
			self.q[ii] = (w[ii]+self.M0f[ii])/hv[ii]
			
		return self.q

	def diagnose_F(self,h,u):
		M1h = self.Uh.assemble(h)
		M1invM1h = self.M1inv*M1h
		F = M1invM1h*u
		return F

	def diagnose_K(self,u):
		WtQU = self.WU.assemble(u)
		K = self.M2inv*WtQU
		k = 0.5*K*u
		return k

	def prognose_h(self,hi,F,dt):
		hf = hi - dt*self.detInv*self.D21*F
		return hf

	def prognose_u(self,hi,ui,hd,ud,q,F,dt):
		# remove the anticipated potential vorticity from q
		if self.apvm > 0.0:
			dq = self.D10*q
			PtQU = self.PU.assemble(dq)
			udq = self.M0inv*PtQU*ud
			q = q - self.apvm*udq

		R = self.Rq.assemble(q)
		qCrossF = R*F
		
		k = self.diagnose_K(ud)
		hBar = k + self.g*hd + self.g*self.hb
		gE = self.D12M2*hBar

		uf = ui - dt*self.M1inv*(qCrossF + gE)

		return uf

	def solveEuler(self,hi,ui,hd,ud,dt):
		q = self.diagnose_q(hd,ud)
		F = self.diagnose_F(hd,ud)

		hf = self.prognose_h(hi,F,dt)
		uf = self.prognose_u(hi,ui,hd,ud,q,F,dt)

		return hf,uf

	def solveRK2(self,hi,ui,dt):
		hh,uh = self.solveEuler(hi,ui,hi,ui,0.5*dt)
		hf,uf = self.solveEuler(hi,ui,hh,uh,dt)

		return hf,uf
