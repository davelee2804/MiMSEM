import numpy as np

from Basis import *

def mult(M, N):
	mi = M.shape[0]
	mj = M.shape[1]
	ni = N.shape[0]
	nj = N.shape[1]

	if mj != ni:
		print 'ERROR! matrix shapes dont match'

	MN = np.zeros((mi,nj),dtype=np.float64)
	for i in np.arange(mi):
		for j in np.arange(nj):
			for k in np.arange(mj):
				MN[i,j] = MN[i,j] + M[i,k]*N[k,j]

	return MN

# Outer product of 0-form in x and 1-form in y (columns)
# evaluated at the Gauss Lobatto quadrature points (rows)
# n: basis function order
# m: quadrature point order
class M1_j_x_i:
	def __init__(self,n,m):
		mp1 = m+1
		mi = mp1
		nj = n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)
		Mj = LagrangeEdge(n)

		for j in np.arange(nj):
			for i in np.arange(mi):
				x = q.x[i]
				Mjx = Mj.eval(x,j)
				self.A[i,j] = Mjx

# 0 form basis function terms (j) evaluated at the
# quadrature points (i)
class M0_j_x_i:
	def __init__(self,n,m):
		np1 = n+1
		mp1 = m+1
		mi = mp1
		nj = np1
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)
		Nj = LagrangeNode(n)

		for j in np.arange(nj):
			for i in np.arange(mi):
				x = q.x[i]
				Njx = Nj.eval(x,j)
				self.A[i,j] = Njx

# Quadrature weights diagonal matrix
class Wii:
	def __init__(self,m):
		mp1 = m+1
		mi = mp1
		mj = mp1
		q = GaussLobatto(m)
		self.A = np.zeros((mi,mj),dtype=np.float64)
		for i in np.arange(mi):
			self.A[i,i] = q.w[i]

class dl_j_x_i:
	def __init__(self,n,m):
		np1 = n+1
		mp1 = m+1
		mi = mp1
		nj = np1
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)
		Nj = LagrangeNode(n)

		for j in np.arange(nj):
			for i in np.arange(mi):
				x = q.x[i]
				Njx = Nj.evalDeriv(x,j)
				self.A[i,j] = Njx

