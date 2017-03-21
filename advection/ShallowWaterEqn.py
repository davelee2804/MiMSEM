import numpy as np
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from Mats2D import *
from Assembly import *
from BoundaryMat import *
from Proj import *
from LieDeriv import *

class SWEqn:
	def __init__(self,topo,quad,topo_q,lx,ly,g):
		self.topo = topo
		self.quad = quad
		self.topo_q = topo_q # topology for the quadrature points as 0 forms
		self.g = g
		det = 0.5*lx/topo.nx
		self.detInv = 1.0/det

		# 1 form matrix inverse
		M1 = Umat(topo,quad).M
		self.M1inv = la.inv(M1)

		# 2 form matrix inverse
		M2 = Wmat(topo,quad).M
		self.M2inv = la.inv(M2)

		# Lie derivative for the mass equation
		self.lie = LieDeriv(topo,quad,lx,ly)

		# 0 form matrix inverse
		M0 = Pmat(topo,quad).M
		M0inv = la.inv(M0)
		D10 = BoundaryMat10(topo).M
		D01 = D10.transpose()
		M0invD01 = M0inv*D01
		self.D01 = self.detInv*M0invD01*M1

		# 2 form gradient matrix
		D21 = BoundaryMat(topo).M
		D12 = D21.transpose()
		#D12M2 = D12*M2
		#self.D12 = self.M1inv*D12M2
		self.D12 = D12*M2

		# Normal to tangent velocity transformation
		self.Utn = UNormToTang(topo,quad)

	def solveRK2(self,u,h,dt):
		uh = np.zeros((u.shape[0]),dtype=np.float64)
		hh = np.zeros((h.shape[0]),dtype=np.float64)
		uf = np.zeros((u.shape[0]),dtype=np.float64)
		hf = np.zeros((h.shape[0]),dtype=np.float64)

		### half step

		# 1.1.1: solve for the 0 form vorticity
		w = self.D01*u
		# 1.1.2: assemble rotational matrix
		R = RotationalMat(self.topo,self.quad,w).M
		# 1.1.3: assemble kinetic energy matrix
		WtQU = WtQUmat(self.topo,self.quad,u).M
		K = self.M2inv*WtQU
		k = 0.5*K*u
		hBar = k + self.g*h

		# 1.2.1: generate the tangent velocities (*u) and rescale ux tangent velocity by -1
		ut = self.Utn.apply(u)
		topo = self.topo
		shift = topo.nx*topo.ny*topo.n*topo.n
		ut[:shift] = -1.0*ut[:shift]

		# 1.3.1: assemble rotational vector
		rhs = R*ut
		# 1.3.2: assemble 2 form gradient vector
		rhs = rhs + self.D12*hBar

		# 1.5.1: update u half step
		uh[:] = u[:] + 0.5*dt*self.detInv*self.M1inv*rhs

		# 1.6.1: assemble lie derivative operator
		duh = self.lie.assemble(u,h,True)
		# 1.6.2: update h half step
		hh[:] = h[:] + 0.5*dt*duh

		### full step
		
		# 2.1.1: solve for the 0 form vorticity
		wh = self.D01*uh
		# 2.1.2: assemble rotational matrix
		R = RotationalMat(self.topo,self.quad,wh).M
		# 2.1.3: assemble kinetic energy matrix
		WtQU = WtQUmat(self.topo,self.quad,uh).M
		K = self.M2inv*WtQU
		kh = 0.5*K*uh
		hBar = kh + self.g*hh

		# 2.2.1: generate the tangent velocities (*u) and rescale ux tangent velocity by -1
		ut = self.Utn.apply(uh)
		topo = self.topo
		shift = topo.nx*topo.ny*topo.n*topo.n
		ut[:shift] = -1.0*ut[:shift]

		# 1.3.1: assemble rotational vector
		rhs = R*ut
		# 1.3.2: assemble 2 form gradient vector
		rhs = rhs + self.D12*hBar

		# 1.5.1: update u half step
		uf[:] = u[:] + dt*self.detInv*self.M1inv*rhs

		# 1.6.1: assemble lie derivative operator
		duf = self.lie.assemble(uh,hh,True)
		# 1.6.2: update h half step
		hf[:] = h[:] + dt*duf

		return uf, hf
		
	def solveEuler(self,u,h,dt):
		uf = np.zeros((u.shape[0]),dtype=np.float64)
		hf = np.zeros((h.shape[0]),dtype=np.float64)

		# 1.1.1: solve for the 0 form vorticity
		w = self.D01*u
		# 1.1.2: assemble rotational matrix
		R = RotationalMat(self.topo,self.quad,w).M
		# 1.1.3: assemble kinetic energy matrix
		WtQU = WtQUmat(self.topo,self.quad,u).M
		K = self.M2inv*WtQU
		k = 0.5*K*u
		hBar = k + self.g*h

		# 1.2.1: generate the tangent velocities (*u) and rescale ux tangent velocity by -1
		ut = self.Utn.apply(u)
		topo = self.topo
		shift = topo.nx*topo.ny*topo.n*topo.n
		ut[:shift] = -1.0*ut[:shift]

		# 1.3.1: assemble rotational vector
		rhs = R*ut
		# 1.3.2: assemble 2 form gradient vector
		rhs = rhs + self.D12*hBar

		# 1.5.1: update u half step
		uf[:] = u[:] + dt*self.detInv*self.M1inv*rhs

		# 1.6.1: assemble lie derivative operator
		duh = self.lie.assemble(u,h,True)
		# 1.6.2: update h half step
		hf[:] = h[:] + dt*duh

		return uf, hf
