import numpy as np
from scipy import sparse

from Basis import *
from Topo import *
from Mats2D import *

# mass matrix for the 1 form vector (x-normal degrees of
# freedom first then y-normal degrees of freedom)
class Umat:
	def __init__(self,topo,quad):
		maps, nnz = self.genMap(topo)
		self.assemble(topo,quad,maps,nnz)
		self.maps = maps

	def assemble(self,topo,quad,maps,nnz):
		Q = Wii(quad.n).A
		U = M1x_j_xy_i(topo.n,quad.n).A
		V = M1y_j_xy_i(topo.n,quad.n).A
		Ut = U.transpose()
		Vt = V.transpose()
		UtQ = mult(Ut,Q)
		VtQ = mult(Vt,Q)
		UtQU = mult(UtQ,U)
		VtQV = mult(VtQ,V)
		
		np1 = topo.n+1
		ncl = np1*topo.n        # number of columns in local matrix, u or v (same as number of rows)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				for jj in np.arange(ncl*ncl):
					row = inds1[jj/ncl]
					col = inds1[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + UtQU[jj/ncl][jj%ncl]

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey)
				for jj in np.arange(ncl*ncl):
					row = inds1[jj/ncl]
					col = inds1[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + VtQV[jj/ncl][jj%ncl]

		nr = 2*topo.nx*topo.ny*topo.n*topo.n
		nc = 2*topo.nx*topo.ny*topo.n*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo):
		np1 = topo.n+1
		ne = np1*topo.n
		nr = topo.nx*topo.ny*2*topo.n*topo.n
		nc = topo.nx*topo.ny*2*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				for jj in np.arange(ne*ne):
					row = inds1[jj/ne]
					col = inds1[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey)
				for jj in np.arange(ne*ne):
					row = inds1[jj/ne]
					col = inds1[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii

# 2 form mass matrix
class Wmat:
	def __init__(self,topo,quad):
		W = M2_j_xy_i(topo.n,quad.n).A
		Wt = W.transpose()
		Q = Wii(quad.n).A
		WtQ = mult(Wt,Q)
		self.Me = mult(WtQ,W)

		n2 = topo.n*topo.n
		n4 = n2*n2
		nnz = topo.nx*topo.ny*n4
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds2 = topo.localToGlobal2(ex,ey)
				for jj in np.arange(n4):
					r = jj/n2
					c = jj%n2
					rows[ii] = inds2[r]
					cols[ii] = inds2[c]
					vals[ii] = self.Me[r][c]
					ii = ii + 1

		nr = topo.nx*topo.ny*n2
		nc = topo.nx*topo.ny*n2
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

# 0 form mass matrix
class Pmat:
	def __init__(self,topo,quad):
		P = M0_j_xy_i(topo.n,quad.n).A
		Pt = P.transpose()
		Q = Wii(quad.n).A
		PtQ = mult(Pt,Q)
		self.Me = mult(PtQ,P)

		maps,nnz = self.genMap(topo)

		n2 = topo.n*topo.n
		np1 = topo.n + 1
		np12 = np1*np1
		np14 = np12*np12
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0 = topo.localToGlobal0(ex,ey)
				for jj in np.arange(np14):
					row = inds0[jj/np12]
					col = inds0[jj%np12]
					ii = maps[row,col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + self.Me[jj/np12,jj%np12]

		nr = topo.nx*topo.ny*n2
		nc = topo.nx*topo.ny*n2
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo):
		np1 = topo.n+1
		ne = np1*np1
		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0 = topo.localToGlobal0(ex,ey)
				for jj in np.arange(ne*ne):
					row = inds0[jj/ne]
					col = inds0[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii

# 1 form mass matrix with 2 forms interpolated to quadrature points
class Uhmat:
	def __init__(self,topo,quad):
		self.topo = topo
		self.quad = quad
		self.maps, self.nnz = self.genMap(topo)
		#self.assemble(topo,quad,maps,nnz,h)

		Q = Wii(quad.n).A
		U = M1x_j_xy_i(topo.n,quad.n).A
		V = M1y_j_xy_i(topo.n,quad.n).A
		self.QU = mult(Q,U)
		self.QV = mult(Q,V)
		
		self.M1x = M1x_j_Fxy_i(topo.n,quad.n)
		self.M1y = M1y_j_Fxy_i(topo.n,quad.n)

	def assemble(self,h):
		topo = self.topo
		quad = self.quad
		maps = self.maps
		nnz = self.nnz
		n2 = topo.n*topo.n
		np1 = topo.n+1
		ncl = np1*topo.n        # number of columns in local matrix, u or v (same as number of rows)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		ck = np.zeros(n2,dtype=np.float64)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)

				for kk in np.arange(n2):
					ck[kk] = h[inds2[kk]]

				U = self.M1x.assemble(ck)
				Ut = U.transpose()
				UtQU = mult(Ut,self.QU)

				for jj in np.arange(ncl*ncl):
					row = inds1[jj/ncl]
					col = inds1[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + UtQU[jj/ncl][jj%ncl]

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)

				for kk in np.arange(n2):
					ck[kk] = h[inds2[kk]]

				V = self.M1y.assemble(ck)
				Vt = V.transpose()
				VtQV = mult(Vt,self.QV)

				for jj in np.arange(ncl*ncl):
					row = inds1[jj/ncl]
					col = inds1[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + VtQV[jj/ncl][jj%ncl]

		nr = 2*topo.nx*topo.ny*topo.n*topo.n
		nc = 2*topo.nx*topo.ny*topo.n*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

		return self.M

	def genMap(self,topo):
		np1 = topo.n+1
		ne = np1*topo.n
		nr = topo.nx*topo.ny*2*topo.n*topo.n
		nc = topo.nx*topo.ny*2*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				for jj in np.arange(ne*ne):
					row = inds1[jj/ne]
					col = inds1[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey)
				for jj in np.arange(ne*ne):
					row = inds1[jj/ne]
					col = inds1[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii

# 0 form mass matrix
class Phmat:
	def __init__(self,topo,quad,h):
		P = M0_j_xy_i(topo.n,quad.n).A
		Q = Wii(quad.n).A
		QP = mult(Q,P)
		M0h = M0_j_Cxy_i(topo.n,quad.n)

		maps,nnz = self.genMap(topo)

		n2 = topo.n*topo.n
		np1 = topo.n + 1
		np12 = np1*np1
		np14 = np12*np12
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		ck = np.zeros(n2,dtype=np.float64)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0 = topo.localToGlobal0(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)
			
				for kk in np.arange(n2):
					ck[kk] = h[inds2[kk]]

				Ph = M0h.assemble(ck)
				Pht = Ph.transpose()
				M0 = mult(Pht,QP)

				for jj in np.arange(np14):
					row = inds0[jj/np12]
					col = inds0[jj%np12]
					ii = maps[row,col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + M0[jj/np12,jj%np12]

		nr = topo.nx*topo.ny*n2
		nc = topo.nx*topo.ny*n2
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo):
		np1 = topo.n+1
		ne = np1*np1
		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0 = topo.localToGlobal0(ex,ey)
				for jj in np.arange(ne*ne):
					row = inds0[jj/ne]
					col = inds0[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii

# assumes inexact integration and a diagonal mass matrix for the 
# 0 form function space (ie: quadrature and basis functions are 
# the same order)
class Phvec:
	def __init__(self,topo,quad):
		self.topo = topo
		self.quad = quad
		n = topo.n
		np1 = n+1

		self.v = np.zeros((topo.nx*topo.ny*topo.n*topo.n),dtype=np.float64)

		edge = LagrangeEdge(n)
		self.E = np.zeros((np1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(np1):
				self.E[i,j] = edge.eval(quad.x[i],j)

	def assemble(self,h):
		topo = self.topo
		quad = self.quad
		n = topo.n
		n2 = n*n
		np1 = n+1
		np12 = np1*np1

		for ii in np.arange(self.v.shape[0]):
			self.v[ii] = 0.0

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0 = topo.localToGlobal0(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)
				for ii in np.arange(np12):
					wt = quad.w[ii/np1]*quad.w[ii%np1]

					ix = ii%np1
					iy = ii/np1
					hj = 0.0
					for jj in np.arange(n2):
						hj = hj + h[inds2[jj]]*self.E[ix,jj%n]*self.E[iy,jj/n]

					self.v[inds0[ii]] = self.v[inds0[ii]] + wt*hj

		return self.v

# Right hand side matrix for the L2 projection of an analytic function
# defined at the quadrature points onto the 2-forms
class WtQmat:
	def __init__(self,topo,quad):
		topo_q = Topo(topo.nx,topo.ny,quad.n)
		# Build the element matrix, assume same for all elements (regular geometry)
		W = M2_j_xy_i(topo.n,quad.n).A
		Wt = W.transpose()
		Q = Wii(quad.n).A
		WtQ = mult(Wt,Q)

		maps, nnz = self.genMap(topo,topo_q)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		np1 = topo.n+1
		mp1 = quad.n+1
		m0 = mp1*mp1
		n2 = topo.n*topo.n
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds2 = topo.localToGlobal2(ex,ey)
				inds0 = topo_q.localToGlobal0(ex,ey)
				for jj in np.arange(m0*n2):
					r = inds2[jj/m0]
					c = inds0[jj%m0]
					ii = maps[r][c]
					rows[ii] = r
					cols[ii] = c
					vals[ii] = vals[ii] + WtQ[jj/m0][jj%m0]
		
		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo_q.n*topo_q.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo,topo_q):
		np1 = topo.n+1
		mp1 = topo_q.n+1
		m0 = mp1*mp1
		n2 = topo.n*topo.n
		nr = topo.nx*topo.ny*topo.n*topo.n       # 2-forms
		nc = topo.nx*topo.ny*topo_q.n*topo_q.n   # 0-forms
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds2 = topo.localToGlobal2(ex,ey)
				inds0 = topo_q.localToGlobal0(ex,ey)
				for jj in np.arange(m0*n2):
					r = inds2[jj/m0]
					c = inds0[jj%m0]
					if maps[r][c] == -1:
						maps[r][c] = ii
						ii = ii + 1

		return maps, ii

class UtQmat:
	def __init__(self,topo,quad):
		topo_q = Topo(topo.nx,topo.ny,quad.n)
		maps,nnz = self.genMap(topo,topo_q)
		self.assemble(topo,topo_q,maps,nnz)

	def assemble(self,topo,topo_q,maps,nnz):
		Q = Wii(topo_q.n).A
		U = M1x_j_xy_i(topo.n,topo_q.n).A
		V = M1y_j_xy_i(topo.n,topo_q.n).A
		Ut = U.transpose()
		Vt = V.transpose()
		UtQ = mult(Ut,Q)
		VtQ = mult(Vt,Q)

		np1 = topo.n+1
		mp1 = topo_q.n+1
		nrl = topo.n*np1  # number of rows in local matrix, (u or v)
		ncl = mp1*mp1     # number of columns in local matrix (quad pts)
		shift1Forms = (topo.n*topo.nx)*(topo.n*topo.ny)
		shift0Forms = (topo_q.n*topo_q.nx)*(topo_q.n*topo_q.ny)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0 = topo_q.localToGlobal0(ex,ey)
				inds1 = topo.localToGlobal1x(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds0[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + UtQ[jj/ncl][jj%ncl]

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0 = topo_q.localToGlobal0(ex,ey) + shift0Forms
				inds1 = topo.localToGlobal1y(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds0[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + VtQ[jj/ncl][jj%ncl]

		nr = 2*topo.nx*topo.ny*topo.n*topo.n
		nc = 2*topo.nx*topo.ny*topo_q.n*topo_q.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo,topo_q):
		np1 = topo.n+1
		mp1 = topo_q.n+1
		nrl = np1*topo.n
		ncl = mp1*mp1
		nr = 2*topo.nx*topo.ny*topo.n*topo.n
		nc = 2*topo.nx*topo.ny*topo_q.n*topo_q.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		shift1Forms = (topo.n*topo.nx)*(topo.n*topo.ny)
		shift0Forms = (topo_q.n*topo_q.nx)*(topo_q.n*topo_q.ny)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0 = topo_q.localToGlobal0(ex,ey)
				inds1 = topo.localToGlobal1x(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds0[jj%ncl]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0 = topo_q.localToGlobal0(ex,ey) + shift0Forms
				inds1 = topo.localToGlobal1y(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds0[jj%ncl]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii

class PtQmat:
	def __init__(self,topo,quad):
		topo_q = Topo(topo.nx,topo.ny,quad.n)
		maps,nnz = self.genMap(topo,topo_q)
		self.assemble(topo,topo_q,maps,nnz)

	def assemble(self,topo,topo_q,maps,nnz):
		Q = Wii(topo_q.n).A
		P = M0_j_xy_i(topo.n,topo_q.n).A
		Pt = P.transpose()
		PtQ = mult(Pt,Q)

		np1 = topo.n+1
		mp1 = topo_q.n+1
		nrl = np1*np1     # number of rows in local matrix, (0 forms)
		ncl = mp1*mp1     # number of columns in local matrix (quad pts)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0q = topo_q.localToGlobal0(ex,ey)
				inds0 = topo.localToGlobal0(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds0[jj/ncl]
					col = inds0q[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + PtQ[jj/ncl][jj%ncl]

		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo_q.n*topo_q.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo,topo_q):
		np1 = topo.n+1
		mp1 = topo_q.n+1
		nrl = np1*np1
		ncl = mp1*mp1
		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo_q.n*topo_q.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0q = topo_q.localToGlobal0(ex,ey)
				inds0 = topo.localToGlobal0(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds0[jj/ncl]
					col = inds0q[jj%ncl]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii

class UtQPmat:
	def __init__(self,topo,quad,u):
		M1x = M1x_j_Exy_i(topo.n,quad.n)
		M1y = M1y_j_Exy_i(topo.n,quad.n)
		Q = Wii(quad.n).A
		W = M0_j_xy_i(topo.n,quad.n).A
		QW = mult(Q,W)
		
		maps,nnz = self.genMap(topo)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		np1 = topo.n+1
		nrl = topo.n*np1
		#ncl = topo.n*topo.n
		ncl = np1*np1
		c = np.zeros((nrl),dtype=np.float64)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				inds0 = topo.localToGlobal0(ex,ey)
				inds1y = topo.localToGlobal1y(ex,ey)

				for kk in np.arange(nrl):
					c[kk] = +1.0*u[inds1y[kk]]

				U = M1x.assemble(c)
				Ut = U.transpose()
				UtQW = mult(Ut,QW)

				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds0[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + UtQW[jj/ncl][jj%ncl]
		
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey)
				inds0 = topo.localToGlobal0(ex,ey)
				inds1x = topo.localToGlobal1x(ex,ey)

				for kk in np.arange(nrl):
					c[kk] = -1.0*u[inds1x[kk]]

				V = M1y.assemble(c)
				Vt = V.transpose()
				VtQW = mult(Vt,QW)

				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds0[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + VtQW[jj/ncl][jj%ncl]

		nr = 2*topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo):
		np1 = topo.n+1
		nrl = topo.n*np1
		#ncl = topo.n*topo.n
		ncl = np1*np1
		nr = 2*topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				inds0 = topo.localToGlobal0(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds0[jj%ncl]
					if maps[row][col] == -1:
						maps[row][col] = ii;
						ii = ii + 1

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey)
				inds0 = topo.localToGlobal0(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds0[jj%ncl]
					if maps[row][col] == -1:
						maps[row][col] = ii;
						ii = ii + 1

		return maps, ii

class WtQUmat:
	def __init__(self,topo,quad):
		self.topo = topo
		self.quad = quad
		Q = Wii(quad.n).A
		self.M1x = M1x_j_Cxy_i(topo.n,quad.n)
		self.M1y = M1y_j_Cxy_i(topo.n,quad.n)
		W = M2_j_xy_i(topo.n,quad.n).A
		Wt = W.transpose()
		self.WtQ = mult(Wt,Q)
		
		self.maps,self.nnz = self.genMap(topo)

	def assemble(self,u):
		topo = self.topo
		maps = self.maps
		nnz = self.nnz
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		np1 = topo.n+1
		nrl = topo.n*topo.n
		ncl = topo.n*np1
		shift = topo.nx*topo.ny*topo.n*topo.n

		cj = np.zeros((ncl),dtype=np.float64)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)

				for kk in np.arange(ncl):
					cj[kk] = u[inds1[kk]]

				U = self.M1x.assemble(cj)
				WtQU = mult(self.WtQ,U)

				for jj in np.arange(nrl*ncl):
					row = inds2[jj/ncl]
					col = inds1[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + WtQU[jj/ncl][jj%ncl]
		
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)

				for kk in np.arange(ncl):
					cj[kk] = u[inds1[kk]]

				V = self.M1y.assemble(cj)
				WtQV = mult(self.WtQ,V)

				for jj in np.arange(nrl*ncl):
					row = inds2[jj/ncl]
					col = inds1[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + WtQV[jj/ncl][jj%ncl]

		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = 2*topo.nx*topo.ny*topo.n*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

		return self.M

	def genMap(self,topo):
		np1 = topo.n+1
		nrl = topo.n*topo.n
		ncl = topo.n*np1
		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = 2*topo.nx*topo.ny*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds2[jj/ncl]
					col = inds1[jj%ncl]
					if maps[row][col] == -1:
						maps[row][col] = ii;
						ii = ii + 1

		shift = topo.nx*topo.ny*topo.n*topo.n
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds2[jj/ncl]
					col = inds1[jj%ncl]
					if maps[row][col] == -1:
						maps[row][col] = ii;
						ii = ii + 1

		return maps, ii

# project the potential vorticity gradient velocity product onto the 0 forms
class PtQUmat:
	def __init__(self,topo,quad):
		self.topo = topo
		Q = Wii(quad.n).A
		self.M1x = M1x_j_Exy_i(topo.n,quad.n)
		self.M1y = M1y_j_Exy_i(topo.n,quad.n)
		P = M0_j_xy_i(topo.n,quad.n).A
		Pt = P.transpose()
		self.PtQ = mult(Pt,Q)
		
		self.maps,self.nnz = self.genMap(topo)

	def assemble(self,dq):
		topo = self.topo
		maps = self.maps
		nnz = self.nnz

		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		np1 = topo.n+1
		nrl = np1*np1
		ncl = topo.n*np1
		shift = topo.nx*topo.ny*topo.n*topo.n

		cj = np.zeros((ncl),dtype=np.float64)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1x = topo.localToGlobal1x(ex,ey)
				inds1y = topo.localToGlobal1y(ex,ey)
				inds0 = topo.localToGlobal0(ex,ey)

				for kk in np.arange(ncl):
					cj[kk] = dq[inds1y[kk]]

				U = self.M1x.assemble(-1.0*cj)
				PtQU = mult(self.PtQ,U)

				for jj in np.arange(nrl*ncl):
					row = inds0[jj/ncl]
					col = inds1x[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + PtQU[jj/ncl][jj%ncl]
		
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1x = topo.localToGlobal1x(ex,ey)
				inds1y = topo.localToGlobal1y(ex,ey)
				inds0 = topo.localToGlobal0(ex,ey)

				for kk in np.arange(ncl):
					cj[kk] = dq[inds1x[kk]]

				V = self.M1y.assemble(cj)
				PtQV = mult(self.PtQ,V)

				for jj in np.arange(nrl*ncl):
					row = inds0[jj/ncl]
					col = inds1y[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + PtQV[jj/ncl][jj%ncl]

		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = 2*topo.nx*topo.ny*topo.n*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

		return self.M

	def genMap(self,topo):
		np1 = topo.n+1
		nrl = np1*np1
		ncl = topo.n*np1
		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = 2*topo.nx*topo.ny*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				inds0 = topo.localToGlobal0(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds0[jj/ncl]
					col = inds1[jj%ncl]
					if maps[row][col] == -1:
						maps[row][col] = ii;
						ii = ii + 1

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey)
				inds0 = topo.localToGlobal0(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds0[jj/ncl]
					col = inds1[jj%ncl]
					if maps[row][col] == -1:
						maps[row][col] = ii;
						ii = ii + 1

		return maps, ii

# Projects 2 form field onto the 1 forms and interpolates
# the velocity there (for the advection equation)
class InteriorProdAdjMat:
	def __init__(self,topo,quad,u):
		Q = Wii(quad.n).A
		W = M2_j_xy_i(topo.n,quad.n).A
		QW = mult(Q,W)
		
		maps,nnz = self.genMap(topo)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		np1 = topo.n+1
		nrl = topo.n*np1
		ncl = topo.n*topo.n
		cj = np.zeros((nrl),dtype=np.float64)

		M1x = M1x_j_Cxy_i(topo.n,quad.n)
		M1y = M1y_j_Cxy_i(topo.n,quad.n)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)

				for j in np.arange(nrl):
					cj[j] = u[inds1[j]]

				# TODO: should u be interpolated onto the quadrature
				# points from the U^T matrix or the W matrix??
				#U = M1x_j_Cxy_i(topo.n,quad.n,cj).A
				U = M1x.assemble(cj)
				Ut = U.transpose()
				UtQW = mult(Ut,QW)

				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds2[jj%ncl]
					ii = maps[row][col]
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + UtQW[jj/ncl][jj%ncl]
		
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)

				for j in np.arange(nrl):
					cj[j] = u[inds1[j]]

				# TODO: should u be interpolated onto the quadrature
				# points from the U^T matrix or the W matrix??
				#V = M1y_j_Cxy_i(topo.n,quad.n,cj).A
				V = M1y.assemble(cj)
				Vt = V.transpose()
				VtQW = mult(Vt,QW)

				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds2[jj%ncl]
					ii = maps[row][col]
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + VtQW[jj/ncl][jj%ncl]

		nr = 2*topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo):
		np1 = topo.n+1
		nrl = topo.n*np1
		ncl = topo.n*topo.n
		nr = 2*topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds2[jj%ncl]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)
				for jj in np.arange(nrl*ncl):
					row = inds1[jj/ncl]
					col = inds2[jj%ncl]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii

# 1 form mass matrix with 0 form interpolated to quadrature points
# (for rotational term in the momentum equation)
class RotationalMat:
	def __init__(self,topo,quad):
		self.topo = topo
		self.quad = quad

		Q = Wii(quad.n).A
		U = M1x_j_xy_i(topo.n,quad.n).A
		V = M1y_j_xy_i(topo.n,quad.n).A
		Ut = U.transpose()
		Vt = V.transpose()
		self.UtQ = mult(Ut,Q)
		self.VtQ = mult(Vt,Q)

		self.M1x = M1x_j_Dxy_i(topo.n,quad.n)
		self.M1y = M1y_j_Dxy_i(topo.n,quad.n)

		self.maps,self.nnz = self.genMap(topo)
	
	def assemble(self,w):
		topo = self.topo
		maps = self.maps
		nnz = self.nnz

		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		n = topo.n
		np1 = topo.n+1
		nrl = topo.n*np1
		ncl = topo.n*np1
		n2 = np1*np1

		cj = np.zeros((n2),dtype=np.float64)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0 = topo.localToGlobal0(ex,ey)
				inds1x = topo.localToGlobal1x(ex,ey)
				inds1y = topo.localToGlobal1y(ex,ey)

				for j in np.arange(n2):
					cj[j] = w[inds0[j]]

				# TODO: should u be interpolated onto the quadrature
				# points from the U^T matrix or the W matrix??
				V = self.M1y.assemble(-1.0*cj)
				UtQV = mult(self.UtQ,V)

				for jj in np.arange(nrl*ncl):
					row = inds1x[jj/ncl]
					col = inds1y[jj%ncl]
					ii = maps[row][col]
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + UtQV[jj/ncl][jj%ncl]
		
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds0 = topo.localToGlobal0(ex,ey)
				inds1y = topo.localToGlobal1y(ex,ey)
				inds1x = topo.localToGlobal1x(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)

				for j in np.arange(n2):
					cj[j] = w[inds0[j]]

				# TODO: should u be interpolated onto the quadrature
				# points from the U^T matrix or the W matrix??
				U = self.M1x.assemble(cj)
				VtQU = mult(self.VtQ,U)

				for jj in np.arange(nrl*ncl):
					row = inds1y[jj/ncl]
					col = inds1x[jj%ncl]
					ii = maps[row][col]
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + VtQU[jj/ncl][jj%ncl]

		nr = 2*topo.nx*topo.ny*topo.n*topo.n
		nc = 2*topo.nx*topo.ny*topo.n*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

		return self.M

	def genMap(self,topo):
		np1 = topo.n+1
		ne = np1*topo.n
		nr = topo.nx*topo.ny*2*topo.n*topo.n
		nc = topo.nx*topo.ny*2*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1x = topo.localToGlobal1x(ex,ey)
				inds1y = topo.localToGlobal1y(ex,ey)
				for jj in np.arange(ne*ne):
					row = inds1x[jj/ne]
					col = inds1y[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1y = topo.localToGlobal1y(ex,ey)
				inds1x = topo.localToGlobal1x(ex,ey)
				for jj in np.arange(ne*ne):
					row = inds1y[jj/ne]
					col = inds1x[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii

class UxxMat:
	def __init__(self,topo,quad):
		maps, nnz = self.genMap(topo)
		self.assemble(topo,quad,maps,nnz)
		self.maps = maps

	def assemble(self,topo,quad,maps,nnz):
		Q = Wii(quad.n).A
		U = M1x_j_xy_i(topo.n,quad.n).A
		Ut = U.transpose()
		UtQ = mult(Ut,Q)
		UtQU = mult(UtQ,U)
		
		np1 = topo.n+1
		ncl = np1*topo.n        # number of columns in local matrix, u or v (same as number of rows)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				for jj in np.arange(ncl*ncl):
					row = inds1[jj/ncl]
					col = inds1[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + UtQU[jj/ncl][jj%ncl]

		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo):
		np1 = topo.n+1
		ne = np1*topo.n
		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1x(ex,ey)
				for jj in np.arange(ne*ne):
					row = inds1[jj/ne]
					col = inds1[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii

class UyyMat:
	def __init__(self,topo,quad):
		maps, nnz = self.genMap(topo)
		self.assemble(topo,quad,maps,nnz)
		self.maps = maps

	def assemble(self,topo,quad,maps,nnz):
		Q = Wii(quad.n).A
		V = M1y_j_xy_i(topo.n,quad.n).A
		Vt = V.transpose()
		VtQ = mult(Vt,Q)
		VtQV = mult(VtQ,V)
		
		np1 = topo.n+1
		ncl = np1*topo.n        # number of columns in local matrix, u or v (same as number of rows)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey) - shift
				for jj in np.arange(ncl*ncl):
					row = inds1[jj/ncl]
					col = inds1[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + VtQV[jj/ncl][jj%ncl]

		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo):
		np1 = topo.n+1
		ne = np1*topo.n
		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1 = topo.localToGlobal1y(ex,ey) - shift
				for jj in np.arange(ne*ne):
					row = inds1[jj/ne]
					col = inds1[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii

class UxyMat:
	def __init__(self,topo,quad):
		maps, nnz = self.genMap(topo)
		self.assemble(topo,quad,maps,nnz)
		self.maps = maps

	def assemble(self,topo,quad,maps,nnz):
		Q = Wii(quad.n).A
		U = M1x_j_xy_i(topo.n,quad.n).A
		V = M1y_j_xy_i(topo.n,quad.n).A
		Ut = U.transpose()
		UtQ = mult(Ut,Q)
		UtQV = mult(UtQ,V)
		
		np1 = topo.n+1
		ncl = np1*topo.n        # number of columns in local matrix, u or v (same as number of rows)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1x = topo.localToGlobal1x(ex,ey)
				inds1y = topo.localToGlobal1y(ex,ey) - shift
				for jj in np.arange(ncl*ncl):
					row = inds1x[jj/ncl]
					col = inds1y[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + UtQV[jj/ncl][jj%ncl]

		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo):
		np1 = topo.n+1
		ne = np1*topo.n
		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1x = topo.localToGlobal1x(ex,ey)
				inds1y = topo.localToGlobal1y(ex,ey) - shift
				for jj in np.arange(ne*ne):
					row = inds1x[jj/ne]
					col = inds1y[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii

class UyxMat:
	def __init__(self,topo,quad):
		maps, nnz = self.genMap(topo)
		self.assemble(topo,quad,maps,nnz)
		self.maps = maps

	def assemble(self,topo,quad,maps,nnz):
		Q = Wii(quad.n).A
		U = M1x_j_xy_i(topo.n,quad.n).A
		V = M1y_j_xy_i(topo.n,quad.n).A
		Vt = V.transpose()
		VtQ = mult(Vt,Q)
		VtQU = mult(VtQ,U)
		
		np1 = topo.n+1
		ncl = np1*topo.n        # number of columns in local matrix, u or v (same as number of rows)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)

		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1x = topo.localToGlobal1x(ex,ey)
				inds1y = topo.localToGlobal1y(ex,ey) - shift
				for jj in np.arange(ncl*ncl):
					row = inds1y[jj/ncl]
					col = inds1x[jj%ncl]
					ii = maps[row][col]
					if ii == -1:
						print 'ERROR! assembly'
					rows[ii] = row
					cols[ii] = col
					vals[ii] = vals[ii] + VtQU[jj/ncl][jj%ncl]

		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo):
		np1 = topo.n+1
		ne = np1*topo.n
		nr = topo.nx*topo.ny*topo.n*topo.n
		nc = topo.nx*topo.ny*topo.n*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		shift = (topo.n*topo.nx)*(topo.n*topo.ny)
		ii = 0
		for ey in np.arange(topo.ny):
			for ex in np.arange(topo.nx):
				inds1x = topo.localToGlobal1x(ex,ey)
				inds1y = topo.localToGlobal1y(ex,ey) - shift
				for jj in np.arange(ne*ne):
					row = inds1y[jj/ne]
					col = inds1x[jj%ne]
					if maps[row][col] == -1:
						maps[row][col] = ii
						ii = ii + 1

		return maps, ii
