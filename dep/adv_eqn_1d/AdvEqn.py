import numpy as np
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from Mats1D import *
from Assembly import *
from BoundaryMat import *
from Proj import *

class AdvEqn:
	def __init__(self,topo,quad,dX,dt,vel):
		self.topo = topo
		self.quad = quad
		self.dX = dX
		self.dt = dt

		# 1 form matrix inverse
		self.M1 = Pmat(topo,quad,dX).M
		self.M1inv = la.inv(self.M1)

		# 0 form matrix inverse
		self.M0 = Umat(topo,quad,dX).M
		self.M0inv = la.inv(self.M0)

		# 1 form gradient matrix
		self.E10 = BoundaryMat(topo).M
		self.E01 = self.E10.transpose()

		# <gamma,u.beta>
		self.M10 = PtU_u(topo,quad,dX,vel).M
		self.M01 = self.M10.transpose()

		# solution operators
		M1E10 = self.M1*self.E10
		M1E10M0inv = M1E10*self.M0inv
		self.A = M1E10M0inv*self.M01
		self.At = -1.0*self.A.transpose()

		self.A2 = 0.5*self.A + 0.5*self.At

		# solution operators for the implicit solve
		self.L = self.M1 + 0.5*dt*self.A2
		self.R = self.M1 - 0.5*dt*self.A2
		self.Linv = la.inv(self.L)
		self.S = self.Linv*self.R

		# implicit solvers for the advective and material forms
		self.La = self.M1 + 0.5*dt*self.A
		self.Ra = self.M1 - 0.5*dt*self.A
		self.La_inv = la.inv(self.La)
		self.Sa = self.La_inv*self.Ra

		self.Lt = self.M1 + 0.5*dt*self.At
		self.Rt = self.M1 - 0.5*dt*self.At

		self.Lt_inv = la.inv(self.Lt)
		self.St = self.Lt_inv*self.Rt

		# upwinded form
		node = LagrangeNode(topo.n,quad.n)
		for i in np.arange(node.m+1):
			for j in np.arange(node.n+1):
				node.M_ij_u[i,j] = node.eval(node.quad.x[i]+dt*vel[0]*(2.0/dX[0]),node.quad.x,j)
				node.M_ij_d[i,j] = node.eval(node.quad.x[i]-dt*vel[0]*(2.0/dX[0]),node.quad.x,j)

		edge = LagrangeEdge(topo.n,quad.n)
		edge.M_ij_u = np.zeros((edge.m+1,edge.n),dtype=np.float64)
		for i in np.arange(edge.m+1):
			for j in np.arange(edge.n):
				for k in np.arange(j+1):
					edge.M_ij_u[i,j] = edge.M_ij_u[i,j] - edge.node.eval_deriv(edge.node.quad.x[i]+dt*vel[0]*(2.0/dX[0]),edge.node.quad.x,k)
					edge.M_ij_d[i,j] = edge.M_ij_u[i,j] - edge.node.eval_deriv(edge.node.quad.x[i]-dt*vel[0]*(2.0/dX[0]),edge.node.quad.x,k)

		# flux form
		M1_c_c = Pmat_up(topo,quad,dX,edge.M_ij_c,edge.M_ij_c).M
		M0_u_c = Umat_up_2(topo,quad,dX,vel,node,dt).M
		M0_u_c_inv = la.inv(M0_u_c)
		M01_up2 = U_u_TP_up_2(topo,quad,dX,vel,node,edge,dt,+1.0).M
		A_up2 = M1_c_c * self.E10 * M0_u_c_inv * M01_up2 # mass conserving

		# material form
		M0_d_c = Umat_up(topo,quad,dX,node.M_ij_d,node.M_ij_c).M
		M0_d_c_inv = la.inv(M0_d_c)
		M01_up2 = U_u_TP_up_2(topo,quad,dX,vel,node,edge,dt,-1.0).M
		A_up2_T = M1_c_c * self.E10 * M0_d_c_inv * M01_up2 # mass conserving
		A_dep = - 1.0*A_up2_T.transpose()

		L = M1_c_c + 0.5*dt*(A_dep) # smoother solution
		#L = M1_c_c + 0.25*dt*(A_dep - A_dep.transpose()) # energy conserving
		L_inv = la.inv(L);
		R = M1_c_c - 0.5*dt*(A_dep) # smoother solution
		#R = M1_c_c - 0.25*dt*(A_dep - A_dep.transpose()) # energy conserving
		self.Q_up = L_inv * R

		L = M1_c_c + 0.5*dt*(A_up2) # smoother solution
		#L = M1_c_c + 0.25*dt*(A_up2 - A_up2.transpose()) # energy conserving
		L_inv = la.inv(L);
		R = M1_c_c - 0.5*dt*(A_up2) # smoother solution
		#R = M1_c_c - 0.25*dt*(A_up2 - A_up2.transpose()) # energy conserving
		self.Q_up_2 = L_inv * R

		self.A_upw = self.M1inv * A_up2
		self.A_cen = self.M1inv * self.A
		self.A_up2 = self.M1inv * A_dep

		# for the convergence tests (F = qu)
		M1_c_c = Pmat_up(topo,quad,dX,edge.M_ij_c,edge.M_ij_c).M
		M1_c_u = Pmat_up(topo,quad,dX,edge.M_ij_c,edge.M_ij_u).M
		M0_c_c = Umat_up(topo,quad,dX,node.M_ij_c,node.M_ij_c).M
		M0_u_c = Umat_up_2(topo,quad,dX,vel,node,dt).M
		M0_c_u = Umat_up(topo,quad,dX,node.M_ij_c,node.M_ij_u).M
		M0_u_u = Umat_up(topo,quad,dX,node.M_ij_u,node.M_ij_u).M
		M0_c_c_inv = la.inv(M0_c_c)
		M0_u_c_inv = la.inv(M0_u_c)
		M0_c_u_inv = la.inv(M0_c_u)
		M0_u_u_inv = la.inv(M0_u_u)
		M01_arr = U_u_TP_up_2(topo,quad,dX,vel,node,edge,dt,+0.0).M
		M01_up2 = U_u_TP_up_2(topo,quad,dX,vel,node,edge,dt,+1.0).M
		self.B     = M0_c_c_inv * M01_arr
		self.B_up2 = M0_u_c_inv * M01_up2

		# for the convergence test (u.GRAD p)
		self.C     = -1.0*self.M1inv*(self.B.transpose())*self.E01*self.M1
		M01_dwn = U_u_TP_up_2(topo,quad,dX,vel,node,edge,dt,-1.0).M
		M01_dwn_T = M01_dwn.transpose()
		M0_d_c = Umat_up_2(topo,quad,dX,vel,node,-dt).M
		M0_d_c_T = M0_d_c.transpose()
		M0_d_c_T_inv = la.inv(M0_d_c_T)
		self.C_up2 = -1.0*self.M1inv*M01_dwn_T*M0_d_c_T_inv*self.E01*self.M1

	def solve_a(self,hi):
		#return self.St*hi
		return self.Sa*hi

	def solve_a_up(self,hi):
		#return self.Q_up*hi
		return self.Q_up_2*hi
		
	def solve_2(self,hi):
		#return self.Q_up_2*hi
		return self.Q_up*hi

	def disp_rel(self,AA,do_real):
		quad = self.quad
		nx = len(self.dX) * quad.n
		x = np.zeros(nx)
		for ii in np.arange(len(self.dX)):
                	x[ii*quad.n:ii*quad.n+quad.n] = ii*self.dX[0] + self.dX[0]*0.5*(quad.x[:quad.n]+1)

		xx = 2.0*np.pi*x
		kk = 1.0*(np.arange(nx) - nx/2 + 1)
		x2 = (2.0*np.pi/nx)*np.arange(nx)

		F = np.zeros((nx,nx),dtype=np.complex128)
		for ii in np.arange(nx):
			for jj in np.arange(nx):
				F[ii,jj] = np.exp(1.0j*kk[jj]*xx[ii])

		Finv = np.linalg.inv(F)

		node = LagrangeNode(self.topo.n,self.quad.n)
		edge = LagrangeEdge(self.topo.n,self.quad.n)
		QP = P_interp(self.topo,self.quad,self.dX,node,edge).M
		PQ = PtQmat(self.topo,self.quad,self.dX).M

		P2F = Finv * QP

		II = np.zeros((nx,nx),dtype=np.complex128)
		np.fill_diagonal(II,+1.0)

		vals,vecs = la.eigs(AA,M=II,k=nx-2)
		if do_real:
			vr = vals.real
			vecr = vecs.real
		else:
			vr = vals.imag
			vecr = vecs.imag
		inds = np.argsort(vr)[::-1]
		vr2 = vr[inds]
		vecr2 = vecr[:,inds]

		ki = np.zeros((nx-2,8),dtype=np.int32)

		for ii in np.arange(nx-2):
			vf = np.dot(P2F,vecr2[:,ii])
			inds = np.argsort(np.abs(vf))[::-1]
			ki[ii] = np.abs(kk[inds[0]])

		return ki,vr2
