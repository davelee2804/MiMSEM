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
class M1x_j_xy_i:
	def __init__(self,n,m):
		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)
		Nj = LagrangeNode(n)
		Mj = LagrangeEdge(n)

		for j in np.arange(nj):
			for i in np.arange(mi):
				x = q.x[i%mp1]
				y = q.x[i/mp1]
				Njx = Nj.eval(x,j%np1)
				Mjy = Mj.eval(y,j/np1)
				self.A[i,j] = Njx*Mjy

# As above but with 1 forms interpolated to quadrature points
# (normal) velocity interpolated to quadrature points
class M1x_j_Cxy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	def assemble(self,c):
		n = self.n
		np1 = self.n+1
		mp1 = self.m+1
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(nj):
				ck = ck + c[k]*self.N[i%mp1,k%np1]*self.E[i/mp1,k/np1]

			for j in np.arange(nj):
				Njx = self.N[i%mp1,j%np1]
				Mjy = self.E[i/mp1,j/np1]

				self.A[i,j] = Njx*Mjy*ck

		return self.A

# As above but with 1 forms cross product interpolated to quadrature points
# (tangent) velocity interpolated to quadrature points
class M1x_j_Exy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	def assemble(self,c):
		n = self.n
		np1 = self.n+1
		mp1 = self.m+1
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(nj):
				ck = ck + c[k]*self.E[i%mp1,k%n]*self.N[i/mp1,k/n]

			for j in np.arange(nj):
				Njx = self.N[i%mp1,j%np1]
				Mjy = self.E[i/mp1,j/np1]

				self.A[i,j] = Njx*Mjy*ck

		return self.A

# As above but with 0 forms interpolated to quadrature points
# potential vorticity interpolated to quadrature points
class M1x_j_Dxy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	#def assemble(self,c,d):
	def assemble(self,c):
		np1 = self.n + 1
		mp1 = self.m + 1
		n = self.n
		n2 = np1*np1
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(n2):
				ck = ck + c[k]*self.N[i%mp1,k%np1]*self.N[i/mp1,k/np1]

			#dk = 0.0
			#for k in np.arange(n*n):
			#	dk = dk + d[k]*self.E[i%mp1,k%n]*self.E[i/mp1,k/n]

			for j in np.arange(nj):
				Njx = self.N[i%mp1,j%np1]
				Mjy = self.E[i/mp1,j/np1]

				self.A[i,j] = Njx*Mjy*ck#/dk

		return self.A

# thickness interpolated to quadrature points
# for diagnosis of hu
class M1x_j_Fxy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	def assemble(self,c):
		np1 = self.n + 1
		mp1 = self.m + 1
		n = self.n
		n2 = n*n
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(n2):
				ck = ck + c[k]*self.E[i%mp1,k%n]*self.E[i/mp1,k/n]

			for j in np.arange(nj):
				Njx = self.N[i%mp1,j%np1]
				Mjy = self.E[i/mp1,j/np1]

				self.A[i,j] = Njx*Mjy*ck

		return self.A

# Outer product of 1-form in x and 0-form in y (columns)
# evaluated at the Gauss Lobatto quadrature points (rows)
# n: basis function order
# m: quadrature point order
class M1y_j_xy_i:
	def __init__(self,n,m):
		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)
		Nj = LagrangeNode(n)
		Mj = LagrangeEdge(n)

		for j in np.arange(nj):
			for i in np.arange(mi):
				x = q.x[i%mp1]
				y = q.x[i/mp1]
				Mjx = Mj.eval(x,j%n)
				Njy = Nj.eval(y,j/n)
				self.A[i,j] = Mjx*Njy

# As above but with 1 forms interpolated to quadrature points
# (normal) velocity interpolated to quadrature points
class M1y_j_Cxy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	def assemble(self,c):
		np1 = self.n+1
		mp1 = self.m+1
		n = self.n
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(nj):
				ck = ck + c[k]*self.E[i%mp1,k%n]*self.N[i/mp1,k/n]

			for j in np.arange(nj):
				Mjx = self.E[i%mp1,j%n]
				Njy = self.N[i/mp1,j/n]

				self.A[i,j] = Mjx*Njy*ck

		return self.A

# As above but with 1 forms (cross product) interpolated to quadrature points
# (tangent) velocity interpolated to quadrature points
class M1y_j_Exy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	def assemble(self,c):
		np1 = self.n+1
		mp1 = self.m+1
		n = self.n
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(nj):
				ck = ck + c[k]*self.N[i%mp1,k%np1]*self.E[i/mp1,k/np1]

			for j in np.arange(nj):
				Mjx = self.E[i%mp1,j%n]
				Njy = self.N[i/mp1,j/n]

				self.A[i,j] = Mjx*Njy*ck

		return self.A

# As above but with 0 forms interpolated to quadrature points
# potential vorticity interpolated to quadrature points
class M1y_j_Dxy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		n2 = n*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)


	#def assemble(self,c,d):
	def assemble(self,c):
		n = self.n
		np1 = self.n + 1
		mp1 = self.m + 1
		n2 = np1*np1
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(n2):
				ck = ck + c[k]*self.N[i%mp1,k%np1]*self.N[i/mp1,k/np1]

			#dk = 0.0
			#for k in np.arange(n*n):
			#	dk = dk + d[k]*self.E[i%mp1,k%n]*self.E[i/mp1,k/n]

			for j in np.arange(nj):
				Mjx = self.E[i%mp1,j%n]
				Njy = self.N[i/mp1,j/n]

				self.A[i,j] = Mjx*Njy*ck#/dk

		return self.A

# thickness interpolated to quadrature points
# for diagnosis of hv
class M1y_j_Fxy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		n2 = n*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)


	def assemble(self,c):
		n = self.n
		np1 = self.n + 1
		mp1 = self.m + 1
		n2 = n*n
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(n2):
				ck = ck + c[k]*self.E[i%mp1,k%n]*self.E[i/mp1,k/n]

			for j in np.arange(nj):
				Mjx = self.E[i%mp1,j%n]
				Njy = self.N[i/mp1,j/n]

				self.A[i,j] = Mjx*Njy*ck

		return self.A

# Outer product of 1-form in x and 1-form in y
# evaluated at the Gauss Lobatto quadrature points
# n: basis function order
# m: quadrature point order
class M2_j_xy_i:
	def __init__(self,n,m):
		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = n*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)
		Mj = LagrangeEdge(n)
		for j in np.arange(nj):
			for i in np.arange(mi):
				x = q.x[i%mp1]
				y = q.x[i/mp1]
				Mjx = Mj.eval(x,j%n)
				Mjy = Mj.eval(y,j/n)
				self.A[i,j] = Mjx*Mjy

# 0 form basis function terms (j) evaluated at the
# quadrature points (i)
class M0_j_xy_i:
	def __init__(self,n,m):
		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*np1
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)
		Nj = LagrangeNode(n)
		for j in np.arange(nj):
			for i in np.arange(mi):
				x = q.x[i%mp1]
				y = q.x[i/mp1]
				Njx = Nj.eval(x,j%np1)
				Njy = Nj.eval(y,j/np1)
				self.A[i,j] = Njx*Njy

class M0_j_Cxy_i:
	def __init__(self,n,m):
		self.n = n
		self.n2 = n*n
		self.np1 = n+1
		self.mp1 = m+1
		self.mi = self.mp1*self.mp1
		self.nj = self.np1*self.np1
		q = GaussLobatto(m)
		self.A = np.zeros((self.mi,self.nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((self.mp1,self.np1),dtype=np.float64)
		for j in np.arange(self.np1):
			for i in np.arange(self.mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((self.mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(self.mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	def assemble(self,c):
		n = self.n
		n2 = self.n2
		mi = self.mi
		nj = self.nj
		mp1 = self.mp1
		np1 = self.np1
		
		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(n2):
				ck = ck + c[k]*self.E[i%mp1,k%n]*self.E[i/mp1,k/n]

			for j in np.arange(nj):
				Njx = self.N[i%mp1,j%np1]
				Njy = self.N[i/mp1,j/np1]
				self.A[i,j] = Njx*Njy*ck

		return self.A

# Quadrature weights diagonal matrix
class Wii:
	def __init__(self,m):
		mp1 = m+1
		mi = mp1*mp1
		mj = mp1*mp1
		q = GaussLobatto(m)
		self.A = np.zeros((mi,mj),dtype=np.float64)
		for i in np.arange(mi):
			self.A[i,i] = q.w[i%mp1]*q.w[i/mp1]
