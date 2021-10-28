import numpy as np
import scipy.sparse.linalg as la
from scipy.sparse import bmat

from Basis import *
from Topo import *
from Mats1D import *
from Assembly import *
from BoundaryMat import *
from Proj import *

class SWEqn:
	def __init__(self,topo,quad,X,dX,dt,grav):
		self.topo = topo
		self.quad = quad
		self.X = X
		self.dX = dX
		self.dt = dt
		self.grav = grav
		self.linear = False
		self.topo_q = Topo(topo.nx,quad.n)

		# 1 form matrix inverse
		self.M1 = Pmat(topo,quad,dX).M
		self.M1inv = la.inv(self.M1)

		# 0 form matrix inverse
		self.M0 = Umat(topo,quad,dX).M
		self.M0inv = la.inv(self.M0)

		# 1 form gradient matrix
		self.E10 = BoundaryMat(topo).M
		self.E01 = -1.0*self.E10.transpose()

		self.PtQ = PtQmat(topo,quad,dX).M
		self.PtQT = PtQmat(topo,quad,dX).M.transpose()
		self.UtQ = UtQmat(topo,quad,dX).M
		self.UtQT = UtQmat(topo,quad,dX).M.transpose()

		# helmholtz operator
		GRAD = 0.5*self.grav*self.dt*self.E01*self.M1
		DIV = 0.5*self.dt*self.M1*self.E10
		HELM = self.M1 - DIV*self.M0inv*GRAD
		self.HELMinv = la.inv(HELM)
		self.DM0inv = DIV
		self.M0invG = self.M0inv*GRAD

	def assemble_J(self,u,h):
                uq = plot_u(u,self.X,self.topo,self.topo_q,self.dX)
                hq = plot_h(h,self.X,self.topo,self.topo_q,self.dX)
                PtU = PtU_u(self.topo,self.quad,self.dX,uq).M
                UtP = PtU.transpose()
                M0h = Uhmat(self.topo,self.quad,self.dX,hq).M
                Juu = self.M0 + 0.5*self.dt*self.E01*PtU
                Juh = 0.5*self.dt*self.grav*self.E01*self.M1
                Jhu = 0.5*self.dt*self.M1*self.E10*self.M0inv*M0h
                Jhh = self.M1 + 0.5*self.dt*self.E10*self.M0inv*UtP
                J = bmat([[Juu,Juh],[Jhu,Jhh]])
                return J

	def diagnose_F(self,u1,u2,h1,h2):
		u1q = plot_u(u1,self.X,self.topo,self.topo_q,self.dX)
		u2q = plot_u(u2,self.X,self.topo,self.topo_q,self.dX)
		Mu1 = PtU_u(self.topo,self.quad,self.dX,u1q).M.transpose()
		Mu2 = PtU_u(self.topo,self.quad,self.dX,u2q).M.transpose()

		if self.linear:
			u1h1 = self.M0*u1
			u1h2 = self.M0*u1
			u2h1 = self.M0*u2
			u2h2 = self.M0*u2
		else:
			u1h1 = Mu1*h1
			u1h2 = Mu1*h2
			u2h1 = Mu2*h1
			u2h2 = Mu2*h2

		rhs = 2.0*u1h1 + u1h2 + u2h1 + 2.0*u2h2
		rhs = (1.0/6.0)*rhs
		F = self.M0inv*rhs

		return F

	def diagnose_Phi(self,u1,u2,h1,h2):
		u1q = plot_u(u1,self.X,self.topo,self.topo_q,self.dX)
		u2q = plot_u(u2,self.X,self.topo,self.topo_q,self.dX)
		Mu1 = PtU_u(self.topo,self.quad,self.dX,u1q).M
		Mu2 = PtU_u(self.topo,self.quad,self.dX,u2q).M

		u1u1 = Mu1*u1
		u1u2 = Mu1*u2
		u2u2 = Mu2*u2

		rhs = u1u1 + u1u2 + u2u2
		rhs = (1.0/6.0)*rhs

		gh = 0.5*self.grav*self.M1*(h1+h2)

		rhs = rhs + gh
		if self.linear:
			Phi = self.M1inv*gh
		else:
			Phi = self.M1inv*rhs

		return Phi

	def residual_u(self,u1,u2,h1,h2):
		Fu = self.M0*(u2-u1)
		Phi = self.diagnose_Phi(u1,u2,h1,h2)
		dPhi = self.E01*self.M1*Phi
		Fu = Fu + self.dt*dPhi
		return Fu

	def residual_h(self,u1,u2,h1,h2):
		Fh = self.M1*(h2-h1)
		F = self.diagnose_F(u1,u2,h1,h2)
		dF = self.E10*F
		MdF = self.M1*dF
		Fh = Fh + self.dt*MdF
		return Fh

	def energy(self,u,h):
		uq = plot_u(u,self.X,self.topo,self.topo_q,self.dX)
		Mu = PtU_u(self.topo,self.quad,self.dX,uq).M
		u2 = Mu*u
		K = 0.5*np.dot(h,u2)
		Mh = self.M1*h
		P = 0.5*self.grav*np.dot(h,Mh)
		return K + P

	def solve(self,u1,h1):
		it = 0
		u2 = np.zeros(self.topo.nx*self.topo.n)
		h2 = np.zeros(self.topo.nx*self.topo.n)
		F = np.zeros(len(u2)+len(h2))
		u2[:] = u1[:]
		h2[:] = h1[:]
		# begin newton iteration
		done = False;
		while not done:
			Fu = self.residual_u(u1,u2,h1,h2)
			Fh = self.residual_h(u1,u2,h1,h2)
			F[:len(u2)] = Fu
			F[len(u2):] = Fh
			J = self.assemble_J(u2,h2)
			#Jinv = la.inv(J)
			#dU = Jinv*F
			#dU,info = la.gmres(J,F,tol=1.0e-12)
			dU = self.gmres(J,F)
			du = dU[:len(u2)]
			dh = dU[len(u2):]
			u2 = u2 - du
			h2 = h2 - dh
			# check the error
			dusq = np.dot(du,du)
			usq = np.dot(u2,u2)
			err_u = np.sqrt(dusq/usq)
			dhsq = np.dot(dh,dh)
			hsq = np.dot(h2,h2)
			err_h = np.sqrt(dhsq/hsq)
			print(str(it) + '\t|du|/|u|: ' + str(err_u) + '\t|dh|/|h|: ' + str(err_h))
			it = it + 1
			if it == 3:
                            done = True
			if err_u < 1.0e-12 and err_h < 1.0e-12:
				done = True

		u1[:] = u2[:]
		h1[:] = h2[:]

		return

	def solve_c(self,u1,h1):
		it = 0
		u2 = np.zeros(self.topo.nx*self.topo.n)
		h2 = np.zeros(self.topo.nx*self.topo.n)
		F = np.zeros(len(u2)+len(h2))
		u2[:] = u1[:]
		h2[:] = h1[:]
		# begin newton iteration
		done = False;
		while not done:
			Fu = -1.0*self.residual_u(u1,u2,h1,h2)
			Fh = -1.0*self.residual_h(u1,u2,h1,h2)
			F[:len(u2)] = Fu
			F[len(u2):] = Fh
			J = self.assemble_J(u2,h2)
			if it == 2:
			    dHu = self.diagnose_F(u1,u2,h1,h2)
			    dHh = self.diagnose_Phi(u1,u2,h1,h2)
			    MdH = np.zeros(len(F))
			    MdH[:len(u2)] = self.M0*dHu
			    MdH[len(u2):] = self.M1*dHh
			    dU = self.gmres_c(J,F,MdH)
			    #a=np.dot(dU,MdH)
			    #print('projection error: ' + str(a))
			else:
			    dU = self.gmres(J,F)
			du = dU[:len(u2)]
			dh = dU[len(u2):]
			u2 = u2 + du
			h2 = h2 + dh
			# check the error
			dusq = np.dot(du,du)
			usq = np.dot(u2,u2)
			err_u = np.sqrt(dusq/usq)
			dhsq = np.dot(dh,dh)
			hsq = np.dot(h2,h2)
			err_h = np.sqrt(dhsq/hsq)
			print(str(it) + '\t|du|/|u|: ' + str(err_u) + '\t|dh|/|h|: ' + str(err_h))
			it = it + 1
			if it == 3:
                            done = True
			if err_u < 1.0e-12 and err_h < 1.0e-12:
				done = True

		u1[:] = u2[:]
		h1[:] = h2[:]

		return

	def gmres(self,A,b):
		max_size = 200
		eps = 1.0e-12
		H = np.zeros((max_size+1,max_size))
		Q = np.zeros((max_size+1,len(b)))

		ro = b# - A*x
		beta = np.linalg.norm(ro,2)

		Q[0,:] = ro/beta
		for kk in range(1,max_size+1):
			v = A*Q[kk-1,:]
			for jj in range(kk):
				H[jj,kk-1] = np.dot(Q[jj,:],v)
				v = v - H[jj,kk-1]*Q[jj,:]

			H[kk,kk-1] = np.linalg.norm(v,2)
			if H[kk,kk-1] > eps:
                            Q[kk,:] = v/H[kk,kk-1]
			else:
                            break

		# solve
		e = np.zeros(kk+1)
		e[0] = beta
		H = H[:kk+1,:kk]
		#y = np.linalg.lstsq(H, e, rcond=None)[0]
		y = np.linalg.lstsq(H, e)[0]
		res = np.matmul(H,y) - e
		err = np.linalg.norm(res,2)
		#	if err < 1.0e-16:
		#		break

			# resize the hessenberg
		#	Htmp = H
		#	H = np.zeros((kk+2,kk+1))
		#	H[:kk+1,:kk] = Htmp

		# build the solution
		x = np.matmul(Q[:kk,:].transpose(),y)
		return x

	def gmres_c(self,A,b,c):
		max_size = 200
		eps = 1.0e-12
		H = np.zeros((max_size+2,max_size))
		Q = np.zeros((max_size+2,len(b)))

		c_norm = c/np.linalg.norm(c,2)
		c2inv = 1.0/np.dot(c,c)

		ro = b
		beta = np.linalg.norm(ro,2)
		#ro = ro - c2inv*np.dot(ro,c)*c
		#beta = np.linalg.norm(ro,2)

		Q[0,:] = ro/beta
		Q[0,:] = Q[0,:] - c2inv*np.dot(Q[0,:],c)*c
		for kk in range(1,max_size+1):
			v = A*Q[kk-1,:]
			#H[0,kk-1] = np.dot(c,v)
			#v = v - c2inv*H[0,kk-1]*c
			for jj in range(kk):
				H[jj,kk-1] = np.dot(Q[jj,:],v)
				v = v - H[jj,kk-1]*Q[jj,:]
				#H[jj+1,kk-1] = np.dot(Q[jj,:],v)
				#v = v - H[jj+1,kk-1]*Q[jj,:]

			#vc = np.dot(c,v)
			#v = v - vc*c
			#H[kk,kk-1] = np.dot(c,v)
			#v = v - c2inv*H[kk,kk-1]*c
			H[kk,kk-1] = c2inv*np.dot(c,v)
			v = v - H[kk,kk-1]*c

			H[kk+1,kk-1] = np.linalg.norm(v,2)
			if H[kk+1,kk-1] > eps:
                            Q[kk,:] = v/H[kk+1,kk-1]
			else:
                            break

		# solve
		#print("gmres its: " + str(kk))
		e = np.zeros(kk+2)
		e[0] = beta
		H = H[:kk+2,:kk]
		y = np.linalg.lstsq(H, e)[0]
		res = np.matmul(H,y) - e
		err = np.linalg.norm(res,2)

		# build the solution
		x = np.matmul(Q[:kk,:].transpose(),y)
		#xc = c2inv*np.dot(x,c)
		#print('projection error: ' + str(xc))
		return x

	# constrained gmres iteration
	# A: linear operator
	# x: initial guess and solution
	# b: right hand side
	# c: constraint
	def gmres_constrained(self,A,x,b,c):
		max_size = 20
		H = np.zeros((2,1))
		Q = np.zeros((max_size,len(x)))
		c2inv = 1.0/np.dot(c,c)
		#print('c2inv: ' + str(c2inv))

		ro = b - A*x
		# orthogonalise with respect to c
		#ro = ro - c2inv*np.dot(ro,c)*c
		#print('\torthoganalising w.r.t. c: ' + str(np.dot(ro,c)))
		beta = np.linalg.norm(ro)
		Q[0,:] = ro/beta
		for kk in np.arange(max_size-1)+1:
			v = A*Q[kk-1,:]
			for jj in np.arange(kk):
				H[jj,kk-1] = np.dot(Q[jj,:],v)
				v = v - H[jj,kk-1]*Q[jj,:]
			# orthogonalise with respect to c
			#v = v - c2inv*np.dot(v,c)*c
			#print(str(kk) + '\torthoganalising w.r.t. c: ' + str(np.dot(v,c)))
			H[kk,kk-1] = np.linalg.norm(v)
			Q[kk,:] = v/H[kk,kk-1]
			#print(str(kk) + '\torthoganalising w.r.t. c: ' + str(np.dot(Q[kk,:],c)))
			#print('GMRES iteration: ' + str(kk))
			#print(H)

			# solve
			e = np.zeros(kk+1)
			e[0] = beta
			y = np.linalg.lstsq(H, e, rcond=None)[0]
			res = np.matmul(H,y) - e
			err = np.linalg.norm(res)
			print(str(kk) + ':\terr: ' + str(err))
			if err < 1.0e-16:
				break

			# resize the hessenberg
			Htmp = H
			H = np.zeros((kk+2,kk+1))
			H[:kk+1,:kk] = Htmp

		# build the solution
		xi = np.matmul(Q[:kk,:].transpose(),y)
		xi = xi - c2inv*np.dot(xi,c)*c
		print(str(kk) + '\torthoganalising w.r.t. c: ' + str(np.dot(xi,c)))
		return x + xi

	def gmres_constrained_2(self,A,x,b,c):
		max_size = 20
		H = np.zeros((2,1))
		Q = np.zeros((max_size,len(x)))
		du = np.zeros(len(x))

		ro = b - A*x
		beta = np.linalg.norm(ro)
		Q[0,:] = ro/beta
		for kk in np.arange(max_size-1)+1:
			v = A*Q[kk-1,:]
			for jj in np.arange(kk):
				H[jj,kk-1] = np.dot(Q[jj,:],v)
				v = v - H[jj,kk-1]*Q[jj,:]
			H[kk,kk-1] = np.linalg.norm(v)
			Q[kk,:] = v/H[kk,kk-1]

			# solve
			e = np.zeros(kk+1)
			e[0] = beta
			#y = np.linalg.lstsq(H, e, rcond=None)[0]
			y = self.solve_lsq(H, Q, e, c, du)
			res = np.matmul(H,y) - e
			err = np.linalg.norm(res)
			print(str(kk) + ':\terr: ' + str(err))
			du = np.matmul(Q[:kk,:].transpose(),y)
			if err < 1.0e-16:
				break

			# resize the hessenberg
			Htmp = H
			H = np.zeros((kk+2,kk+1))
			H[:kk+1,:kk] = Htmp

		# build the solution
		#y = self.solve_lsq(H, Q, e, c, du)
		#du = np.matmul(Q[:kk,:].transpose(),y)
		return x + du

	def solve_lsq(self, H, Q, e, V, du):
		kp1,kk = H.shape
		HT = H.transpose()
		HTH = np.matmul(HT,H)
		QV = np.matmul(Q[:kk,:],V)

		A = np.zeros((kp1,kp1))
		A[:kk,:kk] = HTH
		A[kk,:kk] = QV
		A[:kk,kk] = QV
		b = np.zeros(kp1)
		b[:kk] = np.matmul(HT,e)
		b[kk] = -1.0*np.dot(V,du)
		Ainv = np.linalg.inv(A)
		y = np.matmul(Ainv,b)
		return y[:kk]

		#Ainv = np.linalg.inv(HTH)
		#b = np.matmul(HT,e)
		#return np.matmul(Ainv,b)
