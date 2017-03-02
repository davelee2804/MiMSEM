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

		# Lie derivative for the mass equation
		self.lie = LieDeriv(topo,quad,lx,ly)

		# 0 form matrix inverse
		M0 = Pmat(topo,quad).M
		M0inv = la.inv(M0)
		D10 = BoundaryMat10(topo).M
		D01 = D10.transpose()
		M0invD01 = M0inv*D01
		self.D = M0invD01*M1

		# 2 form gradient matrix
		M2 = Wmat(topo,quad).M
		D21 = BoundaryMat(topo).M
		D12 = D21.transpose()
		self.D12M2 = D12*M2

		# Normal to tangent velocity transformation
		self.Utn = UNormToTang(topo,quad)

	def assembleKEVec(un,ut):
		topo = self.topo
		topo_q = self.topo_q

		n = topo.n
		n2 = n*n
		np1 = n+1
		nqp1 = topo_q.n+1
		nq2 = nqp1*nqp1
		shift = topo.nx*topo.ny*n2

		keq = np.zeros((topo.nx*topo.ny*topo_q.n*topo_q.n),dtype=np.float64)

		node = LagrangeNode(n)

		ux = np.zeros((nqp1,np1),dtype=np.float64)
		for i in np.arange(nqp1):
			for j in np.arange(np1):
				ux[i,j] = node.eval(quad.x[i],j)

		for ex in np.arange(self.topo.nx):
			for ey in np.arange(self.topo.ny):
				inds1x = topo.localToGlobal1x(ex,ey)
				inds1y = topo.localToGlobal1y(ex,ey) + shift
				inds2 = topo.localToGlobal2(ex,ey)

				for ii in np.arange(n2):
					i = ii%n
					j = ii/n
					for qq in np.arange(nq2):
						qx = qq%nqp1
						qy = qq/nqp1

						for k in np.arange(np1):
							for l in np.arange(np1):
								kexk = un[inds1x[j*np1+k]]*ux[qx,k]
								kexl = ut[inds1y[l*n+i]]*ux[qy,l]

								keyl = un[inds1y[l*n+i]]*ux[qy,l]
								keyk = ut[inds1x[j*np1+k]]*ux[qx,k]

								keq[inds2[jj]] = keq[inds2[jj]] + 0.5*kexk*kexl + 0.5*keyl*keyk

		return keq

	def solveRK2(self,u,h,dt):
		uh = np.zeros((u.shape[0]),dtype=np.float64)
		hh = np.zeros((h.shape[0]),dtype=np.float64)
		uf = np.zeros((u.shape[0]),dtype=np.float64)
		hf = np.zeros((h.shape[0]),dtype=np.float64)

		### half step

		# 1.1.1: solve for the 0 form vorticity
		w = self.D*u
		# 1.1.2: assemble rotational matrix
		R = RotationalMat(self.topo,self.quad,w).M

		# 1.2.1: solve for *u (tangent velocity)
		ut = self.Utn.apply(u)
		# 1.2.2: assemble kinetic energy 2 form vector
		k = self.assembleKEVec(u,ut)

		# 1.3.1: rescale the ux tangent velocities by -1
		topo = self.topo
		shift = topo.nx*topo.ny*topo.n*topo.n
		for i in np.arange(shift):
			ut[i] = -1.0*ut[i]
		# 1.3.2: assemble rotational vector
		rhs = R*ut

		# 1.4.1: assemble 2 form gradient vector
		rhs = rhs + self.D12M2inv*(k + self.g*h)

		# 1.5.1: update u half step
		uh[:] = u[:] + 0.5*dt*self.detInv*rhs

		# 1.6.1: assemble lie derivative operator
		duh = self.lie.assemble(u,h,True)
		# 1.6.2: update h half step
		hh[:] = h[:] + 0.5*dt*duh

		### full step
		
		# 1.1.1: solve for the 0 form vorticity
		w = self.D*uh
		# 1.1.2: assemble rotational matrix
		R = RotationalMat(self.topo,self.quad,w).M

		# 1.2.1: solve for *u (tangent velocity)
		ut = self.Utn.apply(uh)
		# 1.2.2: assemble kinetic energy 2 form vector
		k = self.assembleKEVec(uh,ut)

		# 1.3.1: rescale the ux tangent velocities by -1
		for i in np.arange(shift):
			ut[i] = -1.0*ut[i]
		# 1.3.2: assemble rotational vector
		rhs = R*ut

		# 1.4.1: assemble 2 form gradient vector
		rhs = rhs + self.D12M2inv*(k + self.g*hh)

		# 1.5.1: update u half step
		uf[:] = u[:] + 0.5*dt*self.detInv*rhs

		# 1.6.1: assemble lie derivative operator
		duh = self.lie.assemble(uh,hh,True)
		# 1.6.2: update h half step
		hf[:] = h[:] + 0.5*dt*duh

		return uf, hf
		
