import numpy as np
from scipy import sparse

from Basis import *
from Topo import *
from Mats1D import *

# 1 form mass matrix
class Pmat:
	def __init__(self,topo,quad,dX):
		P = M1_j_x_i(topo.n,quad.n).A
		Pt = P.transpose()

		n = topo.n
		n2 = topo.n*topo.n
		nnz = topo.nx*n2
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		ii = 0
		for ex in np.arange(topo.nx):
			inds1 = topo.localToGlobal1(ex)
			Q = (2.0/dX[ex]) * Wii(quad.n).A
			PtQ = mult(Pt,Q)
			self.Me = mult(PtQ,P)
			for jj in np.arange(n2):
				r = jj/n
				c = jj%n
				rows[ii] = inds1[r]
				cols[ii] = inds1[c]
				vals[ii] = self.Me[r,c]
				ii = ii + 1

		nr = topo.nx*n
		nc = topo.nx*n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

# 0 form mass matrix
class Umat:
	def __init__(self,topo,quad,dX):
		P = M0_j_x_i(topo.n,quad.n).A
		Pt = P.transpose()

		maps,nnz = self.genMap(topo)

		n2 = topo.n*topo.n
		np1 = topo.n + 1
		np12 = np1*np1
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		for ex in np.arange(topo.nx):
			inds0 = topo.localToGlobal0(ex)
			Q = (dX[ex]/2.0) * Wii(quad.n).A
			PtQ = mult(Pt,Q)
			self.Me = mult(PtQ,P)
			for jj in np.arange(np12):
				row = inds0[jj/np1]
				col = inds0[jj%np1]
				ii = maps[row,col]
				if ii == -1:
					print 'ERROR! assembly'
				rows[ii] = row
				cols[ii] = col
				vals[ii] = vals[ii] + self.Me[jj/np1,jj%np1]

		nr = topo.nx*topo.n
		nc = topo.nx*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo):
		np1 = topo.n+1
		ne = np1
		nr = topo.nx*topo.n
		nc = topo.nx*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ex in np.arange(topo.nx):
			inds0 = topo.localToGlobal0(ex)
			for jj in np.arange(ne*ne):
				row = inds0[jj/ne]
				col = inds0[jj%ne]
				if maps[row][col] == -1:
					maps[row][col] = ii
					ii = ii + 1

		return maps, ii

# Right hand side matrix for the L2 projection of an analytic function
# defined at the quadrature points onto the 1-forms
class PtQmat:
	def __init__(self,topo,quad,dX):
		topo_q = Topo(topo.nx,quad.n)
		# Build the element matrix, assume same for all elements (regular geometry)
		P = M1_j_x_i(topo.n,quad.n).A
		Pt = P.transpose()
		Q = Wii(quad.n).A
		PtQ = mult(Pt,Q)

		maps, nnz = self.genMap(topo,topo_q)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		n = topo.n
		mp1 = quad.n+1
		m0 = mp1
		for ex in np.arange(topo.nx):
			inds1 = topo.localToGlobal1(ex)
			inds0 = topo_q.localToGlobal0(ex)
			for jj in np.arange(m0*n):
				r = inds1[jj/m0]
				c = inds0[jj%m0]
				ii = maps[r][c]
				rows[ii] = r
				cols[ii] = c
				vals[ii] = vals[ii] + PtQ[jj/m0,jj%m0]

		nr = topo.nx*topo.n
		nc = topo.nx*topo_q.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo,topo_q):
		mp1 = topo_q.n+1
		m0 = mp1
		nr = topo.nx*topo.n     # 1-forms
		nc = topo.nx*topo_q.n   # 0-forms (quadrature points)
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ex in np.arange(topo.nx):
			inds1 = topo.localToGlobal1(ex)
			inds0 = topo_q.localToGlobal0(ex)
			for jj in np.arange(m0*topo.n):
				r = inds1[jj/m0]
				c = inds0[jj%m0]
				if maps[r][c] == -1:
					maps[r][c] = ii
					ii = ii + 1

		return maps, ii

class UtQmat:
	def __init__(self,topo,quad,dX):
		topo_q = Topo(topo.nx,quad.n)
		maps,nnz = self.genMap(topo,topo_q)
		self.assemble(topo,topo_q,maps,nnz,dX)

	def assemble(self,topo,topo_q,maps,nnz,dX):
		P = M0_j_x_i(topo.n,topo_q.n).A
		Pt = P.transpose()
		#Q = Wii(topo_q.n).A
		#PtQ = mult(Pt,Q)

		np1 = topo.n+1
		mp1 = topo_q.n+1
		nrl = np1     # number of rows in local matrix, (0 forms)
		ncl = mp1     # number of columns in local matrix (quad pts)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		for ex in np.arange(topo.nx):
			inds0q = topo_q.localToGlobal0(ex)
			inds0 = topo.localToGlobal0(ex)
			Q = (dX[ex]/2.0) * Wii(topo_q.n).A
			PtQ = mult(Pt,Q)
			for jj in np.arange(nrl*ncl):
				row = inds0[jj/ncl]
				col = inds0q[jj%ncl]
				ii = maps[row][col]
				if ii == -1:
					print 'ERROR! assembly'
				rows[ii] = row
				cols[ii] = col
				vals[ii] = vals[ii] + PtQ[jj/ncl][jj%ncl]

		nr = topo.nx*topo.n
		nc = topo.nx*topo_q.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo,topo_q):
		np1 = topo.n+1
		mp1 = topo_q.n+1
		nrl = np1
		ncl = mp1
		nr = topo.nx*topo.n
		nc = topo.nx*topo_q.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ex in np.arange(topo.nx):
			inds0q = topo_q.localToGlobal0(ex)
			inds0 = topo.localToGlobal0(ex)
			for jj in np.arange(nrl*ncl):
				row = inds0[jj/ncl]
				col = inds0q[jj%ncl]
				if maps[row][col] == -1:
					maps[row][col] = ii
					ii = ii + 1

		return maps, ii

# 0 form mass matrix (upwinded)
class Umat_up:
	def __init__(self,topo,quad,dX,M_test,M_trial):
		M_test_T = M_test.transpose()

		maps,nnz = self.genMap(topo)

		n2 = topo.n*topo.n
		np1 = topo.n + 1
		np12 = np1*np1
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		for ex in np.arange(topo.nx):
			inds0 = topo.localToGlobal0(ex)
			Q = (dX[ex]/2.0) * Wii(quad.n).A
			PtQ = mult(M_test_T,Q)
			self.Me = mult(PtQ,M_trial)
			for jj in np.arange(np12):
				row = inds0[jj/np1]
				col = inds0[jj%np1]
				ii = maps[row,col]
				if ii == -1:
					print 'ERROR! assembly'
				rows[ii] = row
				cols[ii] = col
				vals[ii] = vals[ii] + self.Me[jj/np1,jj%np1]

		nr = topo.nx*topo.n
		nc = topo.nx*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def genMap(self,topo):
		np1 = topo.n+1
		ne = np1
		nr = topo.nx*topo.n
		nc = topo.nx*topo.n
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ex in np.arange(topo.nx):
			inds0 = topo.localToGlobal0(ex)
			for jj in np.arange(ne*ne):
				row = inds0[jj/ne]
				col = inds0[jj%ne]
				if maps[row][col] == -1:
					maps[row][col] = ii
					ii = ii + 1

		return maps, ii

# 1 form mass matrix (upwinded)
class Pmat_up:
	def __init__(self,topo,quad,dX,M_test,M_trial):
		M_test_T = M_test.transpose()

		n = topo.n
		n2 = topo.n*topo.n
		nnz = topo.nx*n2
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		ii = 0
		for ex in np.arange(topo.nx):
			inds1 = topo.localToGlobal1(ex)
			Q = (2.0/dX[ex]) * Wii(quad.n).A
			PtQ = mult(M_test_T,Q)
			self.Me = mult(PtQ,M_trial)
			for jj in np.arange(n2):
				r = jj/n
				c = jj%n
				rows[ii] = inds1[r]
				cols[ii] = inds1[c]
				vals[ii] = self.Me[r,c]
				ii = ii + 1

		nr = topo.nx*n
		nc = topo.nx*n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

# test functions are 1 forms and trial functions are 0 forms
class PtU_u:
	def __init__(self,topo,quad,dX,vel_q):
		topo_q = Topo(topo.nx,quad.n)
		P = M1_j_x_i(topo.n,quad.n).A
		U = M0_j_x_i(topo.n,quad.n).A
		Pt = P.transpose()

		U_q = np.zeros((quad.n+1,quad.n+1),dtype=np.float64)

		maps,nnz = self.genMap(topo)
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)

		n = topo.n
		np1 = n+1
		mp1 = quad.n+1
		m0 = mp1
		for ex in np.arange(topo.nx):
			inds1 = topo.localToGlobal1(ex)
			inds0 = topo.localToGlobal0(ex)
			inds0q = topo_q.localToGlobal0(ex)

			# jacobian determinants cancel
			Q = Wii(quad.n).A
			np.fill_diagonal(U_q,vel_q[ex*quad.n:ex*quad.n+quad.n:1])
			PtQ = mult(Pt,Q)
			PtQu = mult(PtQ,U_q)
			PtQuU = mult(PtQu,U)

			for jj in np.arange(np1*n):
				r = inds1[jj/np1]
				c = inds0[jj%np1]
				ii = maps[r][c]
				rows[ii] = r
				cols[ii] = c
				vals[ii] = vals[ii] + PtQuU[jj/np1,jj%np1]

		nr = topo.nx*topo.n
		nc = topo.nx*topo.n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)
		
	def genMap(self,topo):
		mp1 = topo.n+1
		m0 = mp1
		nr = topo.nx*topo.n   # 1-forms
		nc = topo.nx*topo.n   # 0-forms
		maps = -1*np.ones((nr,nc),dtype=np.int32)
		ii = 0
		for ex in np.arange(topo.nx):
			inds1 = topo.localToGlobal1(ex)
			inds0 = topo.localToGlobal0(ex)
			for jj in np.arange(m0*topo.n):
				r = inds1[jj/m0]
				c = inds0[jj%m0]
				if maps[r][c] == -1:
					maps[r][c] = ii
					ii = ii + 1

		return maps, ii

class U_u_TP_up:
	def __init__(self,topo,quad,dX,vel_q,node,edge,dt,step):
		ne = len(dX)
		lx = np.sum(dX)
		nr = ne*node.n
		nc = ne*edge.n
		A = np.zeros((nr,nc),dtype=np.float64)
		B = np.zeros((nr,nc),dtype=np.int32)

		gl = GaussLobatto(node.n)
		for el_i in np.arange(ne):
			for node_i in np.arange(node.n+1):
				for quad_i in np.arange(gl.n+1):
					quad_g = el_i*gl.n+quad_i
					if quad_g == ne*gl.n:
						quad_g = 0

					nq = node.eval(gl.x[quad_i],gl.x,node_i)

					xq = gl.x[quad_i]
					el_j = el_i
					if np.abs(step) > 1.0e-6:
						xq,el_j = self.advect_q(gl.x[quad_i],dX,vel_q[0]*step,dt,el_i,lx)
					for edge_i in np.arange(edge.n):
						eq = edge.eval(xq,edge_i)
						val = nq * gl.w[quad_i] * eq
						row_i = el_i*node.n+node_i
						if row_i == nr:
							row_i = 0
						col_i = el_j*edge.n+edge_i
						#A[row_i,col_i] = A[row_i,col_i] + 0.5*dt*vel_q[0]*val
						A[row_i,col_i] = A[row_i,col_i] + vel_q[0]*val
						B[row_i,col_i] = 1

		nnz = 0
		rows = np.zeros(nr*nc,dtype=np.int32)
		cols = np.zeros(nr*nc,dtype=np.int32)
		vals = np.zeros(nr*nc,dtype=np.float64)
		for row_i in np.arange(nr):
			for col_i in np.arange(nc):
				if B[row_i][col_i] == 1:
					rows[nnz] = row_i
					cols[nnz] = col_i
					vals[nnz] = A[row_i][col_i]
					nnz = nnz + 1

		rows = rows[:nnz]
		cols = cols[:nnz]
		vals = vals[:nnz]

		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def advect_q(self,x,dx,u,dt,ei,lx):
		xo = ei*dx[0] + 0.5*dx[0]*(x+1.0)
		xg = xo - dt*u
		if xg > lx:
			xg = xg - lx
		if xg < 0.0:
			xg = xg + lx
		eq = int(xg/dx[0])
		xq = (xg - eq*dx[0])*(2.0/dx[0]) - 1.0
		if np.abs(xq) > 1.0:
			print 'ERROR! ' + str(xq)
		return xq,eq

class U_u_TP_up_2:
	def __init__(self,topo,quad,dX,vel_q,node,edge,dt,step):
		ne = len(dX)
		lx = np.sum(dX)
		nr = ne*node.n
		nc = ne*edge.n
		A = np.zeros((nr,nc),dtype=np.float64)
		B = np.zeros((nr,nc),dtype=np.int32)

		gl = GaussLobatto(node.n)
		for el_i in np.arange(ne):
			for node_i in np.arange(node.n+1):
				for quad_i in np.arange(gl.n+1):
					nq = node.eval(gl.x[quad_i]+step*(2.0/dX[0])*dt*vel_q[0],gl.x,node_i)
					xq = gl.x[quad_i]
					el_j = el_i

					for edge_i in np.arange(edge.n):
						eq = edge.eval(xq,edge_i)
						#val = nq * gl.w[quad_i] * 0.5*dt*vel_q[0] * eq
						val = nq * gl.w[quad_i] * vel_q[0] * eq
						row_i = el_i*node.n+node_i
						if row_i == nr:
							row_i = 0
						col_i = el_j*edge.n+edge_i
						A[row_i,col_i] = A[row_i,col_i] + val
						B[row_i,col_i] = 1

		nnz = 0
		rows = np.zeros(nr*nc,dtype=np.int32)
		cols = np.zeros(nr*nc,dtype=np.int32)
		vals = np.zeros(nr*nc,dtype=np.float64)
		for row_i in np.arange(nr):
			for col_i in np.arange(nc):
				if B[row_i][col_i] == 1:
					rows[nnz] = row_i
					cols[nnz] = col_i
					vals[nnz] = A[row_i][col_i]
					nnz = nnz + 1

		rows = rows[:nnz]
		cols = cols[:nnz]
		vals = vals[:nnz]

		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def advect_q(self,x,dx,u,dt,ei,lx):
		xo = ei*dx[0] + 0.5*dx[0]*(x+1.0)
		xg = xo - dt*u
		if xg > lx:
			xg = xg - lx
		if xg < 0.0:
			xg = xg + lx
		eq = int(xg/dx[0])
		xq = (xg - eq*dx[0])*(2.0/dx[0]) - 1.0
		if np.abs(xq) > 1.0:
			print 'ERROR! ' + str(xq)
		return xq,eq

class U_u_TP_up_3:
	def __init__(self,topo,quad,dX,vel_q,node,edge,dt,step):
		ne = len(dX)
		lx = np.sum(dX)
		nr = ne*node.n
		nc = ne*edge.n
		A = np.zeros((nr,nc),dtype=np.float64)
		B = np.zeros((nr,nc),dtype=np.int32)

		gl = GaussLobatto(node.n)
		for el_i in np.arange(ne):
			for node_i in np.arange(node.n+1):
				for quad_i in np.arange(gl.n+1):
					nq = node.eval(gl.x[quad_i]+step*(2.0/dX[0])*dt*1.0,gl.x,node_i)
					xq = gl.x[quad_i]
					el_j = el_i

					for edge_i in np.arange(edge.n):
						eq = edge.eval(xq,edge_i)
						val = nq * gl.w[quad_i] * vel_q[0] * eq
						row_i = el_i*node.n+node_i
						if row_i == nr:
							row_i = 0
						col_i = el_j*edge.n+edge_i
						A[row_i,col_i] = A[row_i,col_i] + val
						B[row_i,col_i] = 1

		nnz = 0
		rows = np.zeros(nr*nc,dtype=np.int32)
		cols = np.zeros(nr*nc,dtype=np.int32)
		vals = np.zeros(nr*nc,dtype=np.float64)
		for row_i in np.arange(nr):
			for col_i in np.arange(nc):
				if B[row_i][col_i] == 1:
					rows[nnz] = row_i
					cols[nnz] = col_i
					vals[nnz] = A[row_i][col_i]
					nnz = nnz + 1

		rows = rows[:nnz]
		cols = cols[:nnz]
		vals = vals[:nnz]

		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

	def advect_q(self,x,dx,u,dt,ei,lx):
		xo = ei*dx[0] + 0.5*dx[0]*(x+1.0)
		xg = xo - dt*u
		if xg > lx:
			xg = xg - lx
		if xg < 0.0:
			xg = xg + lx
		eq = int(xg/dx[0])
		xq = (xg - eq*dx[0])*(2.0/dx[0]) - 1.0
		if np.abs(xq) > 1.0:
			print 'ERROR! ' + str(xq)
		return xq,eq

class P_interp:
	def __init__(self,topo,quad,dX,node,edge):
		ne = len(dX)
		nr = ne*node.n
		nc = ne*edge.n
		A = np.zeros((nr,nc),dtype=np.float64)
		B = np.zeros((nr,nc),dtype=np.int32)

		gl = GaussLobatto(node.n)
		for el_i in np.arange(ne):
			for quad_i in np.arange(gl.n+1):
				xq = gl.x[quad_i]
				for edge_i in np.arange(edge.n):
					eq = edge.eval(xq,edge_i)
					val = gl.w[quad_i] * eq
					row_i = el_i*node.n+quad_i
					if row_i == nr:
						row_i = 0
					col_i = el_i*edge.n+edge_i
					A[row_i,col_i] = A[row_i,col_i] + val
					B[row_i,col_i] = 1

		nnz = 0
		rows = np.zeros(nr*nc,dtype=np.int32)
		cols = np.zeros(nr*nc,dtype=np.int32)
		vals = np.zeros(nr*nc,dtype=np.float64)
		for row_i in np.arange(nr):
			for col_i in np.arange(nc):
				if B[row_i][col_i] == 1:
					rows[nnz] = row_i
					cols[nnz] = col_i
					vals[nnz] = A[row_i][col_i]
					nnz = nnz + 1

		rows = rows[:nnz]
		cols = cols[:nnz]
		vals = vals[:nnz]

		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

