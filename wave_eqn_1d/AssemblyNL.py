import numpy as np
from scipy import sparse

from Basis import *
from Topo import *
from Mats1D import *

# 0 form mass matrices with 1 forms interpolated onto quadrature points
class U_pi_mat:
	def __init__(self,topo,quad,edge,pi):
		P = M0_j_x_i(topo.n,quad.n).A
		Pt = P.transpose()
		Q = Wii(quad.n).A
		PtQ = mult(Pt,Q)

		maps,nnz = self.genMap(topo)

		n2 = topo.n*topo.n
		np1 = topo.n + 1
		np12 = np1*np1
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		for ex in np.arange(topo.nx):
			inds0 = topo.localToGlobal0(ex)
			inds1 = topo.localToGlobal1(ex)

			# Q is to be multiplied by J, but pi is to be divided by J,
			# so the J terms cancel and are not assembled into the matrix
			Pq = np.diag(np.dot(edge.ejxi,pi[inds1]))
			A = mult(PtQ,Pq)
			B = mult(A,P)

			for jj in np.arange(np12):
				row = inds0[jj/np1]
				col = inds0[jj%np1]
				ii = maps[row,col]
				if ii == -1:
					print 'ERROR! assembly'
				rows[ii] = row
				cols[ii] = col
				vals[ii] = vals[ii] + B[jj/np1,jj%np1]

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

# 0 form mass matrices with 0 forms interpolated to quadraure points
# (for derivatation of kinetic energy)
class U_vel_mat:
	def __init__(self,topo,quad,vel):
		P = M0_j_x_i(topo.n,quad.n).A
		Pt = P.transpose()
		Q = Wii(quad.n).A
		PtQ = mult(Pt,Q)

		maps,nnz = self.genMap(topo)

		n2 = topo.n*topo.n
		np1 = topo.n + 1
		np12 = np1*np1
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		for ex in np.arange(topo.nx):
			inds0 = topo.localToGlobal0(ex)

			# TODO: weight quadrature points by jacobian determinants?
			# note: this assumes a diagonal mass matrix for the 0 forms
			Pq = vel[inds0]
			A = mult(PtQ,Pq)
			B = mult(A,P)

			for jj in np.arange(np12):
				row = inds0[jj/np1]
				col = inds0[jj%np1]
				ii = maps[row,col]
				if ii == -1:
					print 'ERROR! assembly'
				rows[ii] = row
				cols[ii] = col
				vals[ii] = vals[ii] + B[jj/np1,jj%np1]

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

# 1 form mass matrix with 1 form field interpolated to quadrature points
# note that the 1 form field is in P_1 in the vertical, so must be integrated
# down the layer
class P_pres_mat:
	def __init__(self,topo,quad,dEta,p_kmh,p_kph,do_half):
		P = M1_j_x_i(topo.n,quad.n).A
		Pt = P.transpose()
		Q = Wii(quad.n).A
		PtQ = mult(Pt,Q)

		n = topo.n
		n2 = topo.n*topo.n
		nnz = topo.nx*n2
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		ii = 0
		for ex in np.arange(topo.nx):
			inds1 = topo.localToGlobal1(ex)

			pq_kmh = np.diag(np.dot(edge.ejxi,p_kmh[inds1]))
			pq_kph = np.diag(np.dot(edge.ejxi,p_kph[inds1]))
			if(do_full):
				pk = 0.500*dEta*(1.0*pq_kmh + 1.0*pq_kph)
			else:
				pk = 0.125*dEta*(3.0*pq_kmh + 1.0*pq_kph)

			A = mult(PtQ,pk)
			B = mult(A,P)

			for jj in np.arange(n2):
				r = jj/n
				c = jj%n
				rows[ii] = inds1[r]
				cols[ii] = inds1[c]
				vals[ii] = vals[ii] + B[r,c]
				ii = ii + 1

		nr = topo.nx*n
		nc = topo.nx*n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

# 1 form mass matrix with tau/p interpolated to quadrature points
# note that p is in P_1 in the vertical
# this is the standard approach for ensuring compatability
class P_pres_mat_orig:
	def __init__(self,topo,quad,dEta,p_kmh,p_kph):
		P = M1_j_x_i(topo.n,quad.n).A
		Pt = P.transpose()
		Q = Wii(quad.n).A
		PtQ = mult(Pt,Q)

		n = topo.n
		n2 = topo.n*topo.n
		nnz = topo.nx*n2
		rows = np.zeros(nnz,dtype=np.int32)
		cols = np.zeros(nnz,dtype=np.int32)
		vals = np.zeros(nnz,dtype=np.float64)
		ii = 0
		for ex in np.arange(topo.nx):
			inds1 = topo.localToGlobal1(ex)

			pq = np.zeros((quad.n+1),dtype=np.float64)
			pq_kmh = np.dot(edge.ejxi,p_kmh[inds1])
			pq_kph = np.dot(edge.ejxi,p_kph[inds1])
			for qi in np.arange(quad.n+1):
				pq[qi] = dEta/(0.5*(pq_kmh[qi] + pq_kph[qi]))

			Pq = np.diag(pq)

			A = mult(PtQ,Pq)
			B = mult(A,P)

			for jj in np.arange(n2):
				r = jj/n
				c = jj%n
				rows[ii] = inds1[r]
				cols[ii] = inds1[c]
				vals[ii] = vals[ii] + B[r,c]
				ii = ii + 1

		nr = topo.nx*n
		nc = topo.nx*n
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.float64)

