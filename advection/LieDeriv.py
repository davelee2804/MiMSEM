import numpy as np
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from Mats2D import *
from Assembly import *
from BoundaryMat import *
from Proj import *

# return the 2 form div(uw)
# see equations (71) and (72) of Kreeft, Palha and Gerritsma (2010)
#class LieDeriv:
#	def __init__(self,topo):
#		np1 = topo.n+1
#		self.dq = np.zeros(topo.nx*topo.n*topo.ny*topo.n,dtype=np.float64)
#		self.qx = np.zeros((topo.n,np1),dtype=np.float64)
#		self.qy = np.zeros((np1,topo.n),dtype=np.float64)
#		self.edge = LagrangeEdge(topo.n)
#		self.quad = GaussLobatto(topo.n)
#		self.topo = topo

#	def assemble(self,u,w):
#		n = self.topo.n
#		np1 = n+1
#		nx = self.topo.nx
#		ny = self.topo.ny
#		shift = nx*n*ny*n
#		topo = self.topo
#		qx = np.zeros(2*topo.nx*topo.n*topo.ny*topo.n)

#		for ey in np.arange(ny):
#			for ex in np.arange(nx):
#				inds1x = self.topo.localToGlobal1x(ex,ey)
#				inds1y = self.topo.localToGlobal1y(ex,ey) + shift
#				inds2 = self.topo.localToGlobal2(ex,ey)
				# inner product of vorticity and x velocities
#				for jj in np.arange(n):
#					for ii in np.arange(np1):
#						xl = self.quad.x[ii]
#						self.qx[jj,ii] = 0.0
#						for kk in np.arange(n):
#							self.qx[jj,ii] = self.qx[jj,ii] + w[inds2[jj*n+kk]]*self.edge.eval(xl,kk)

#						self.qx[jj,ii] = u[inds1x[jj*np1+ii]]*self.qx[jj,ii]
#						qx[inds1x[jj*np1+ii]] = self.qx[jj][ii]

				# inner product of vorticity and y velocities
#				for jj in np.arange(np1):
#					for ii in np.arange(n):
#						yl = self.quad.x[jj]
#						self.qy[jj,ii] = 0.0
#						for ll in np.arange(n):
#							self.qy[jj,ii] = self.qy[jj,ii] + w[inds2[ll*n+ii]]*self.edge.eval(yl,ll)

#						self.qy[jj,ii] = u[inds1y[jj*n+ii]]*self.qy[jj,ii]
#						qx[inds1y[jj*n+ii]] = self.qy[jj][ii]

#				for jj in np.arange(n):
#					for ii in np.arange(n):
#						self.dq[inds2[jj*n+ii]] = self.qx[jj,ii+1] - self.qx[jj,ii] + self.qy[jj+1,ii] - self.qy[jj,ii]

#		return self.dq, qx

# Lie derivative via interior product adjoint relation
class LieDeriv:
	def __init__(self,topo,quad,lx,ly):
		self.topo = topo
		self.quad = quad

		M1 = Umat(topo,quad).M
		self.M1inv = la.inv(M1)

		self.Utn = UNormToTang(topo,quad)

		self.initBndry()

		det = 0.5*lx/topo.nx#*0.5*ly/topo.ny
		self.detInv = 1.0/det

	def initBndry(self):
		ne = self.topo.nx*self.topo.ny
		n = self.topo.n
		np1 = n + 1
		n2 = n*n
		shift = ne*n2
		rows = np.zeros((4*shift),dtype=np.int32)
		cols = np.zeros((4*shift),dtype=np.int32)
		vals = np.zeros((4*shift),dtype=np.int8)
		jj = 0
		for el in np.arange(ne):
			ex = el%self.topo.nx
			ey = el/self.topo.nx
			inds1x = self.topo.localToGlobal1x(ex,ey)
			inds1y = self.topo.localToGlobal1y(ex,ey) + shift
			inds2 = self.topo.localToGlobal2(ex,ey)
			for ii in np.arange(n2):
				ix = ii%n
				iy = ii/n
				ib = iy*n + ix
				it = (iy+1)*n + ix
				il = iy*np1 + ix
				ir = iy*np1 + ix + 1

				rows[jj+0] = inds2[ii]
				cols[jj+0] = inds1x[il]
				vals[jj+0] = -1

				rows[jj+1] = inds2[ii]
				cols[jj+1] = inds1x[ir]
				vals[jj+1] = +1

				rows[jj+2] = inds2[ii]
				cols[jj+2] = inds1y[ib]
				vals[jj+2] = -1

				rows[jj+3] = inds2[ii]
				cols[jj+3] = inds1y[it]
				vals[jj+3] = +1

				jj = jj + 4
		
		nr = shift
		nc = 2*shift
		self.D = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.int8)

	def assemble(self,u,q):
		# Project normal to tangent velocities
		ut = self.Utn.apply(u)

		# Assemble the interior product adjoint matrix (for the tangent velocities)
		Iu = InteriorProdAdjMat(self.topo,self.quad,ut).M
		M = self.M1inv*Iu
		uq = M*q

		# Project back to normal components to calculate divergence
		uqn = self.Utn.apply(uq)

		duq = self.detInv*self.D*uqn

		return duq#, uqn
