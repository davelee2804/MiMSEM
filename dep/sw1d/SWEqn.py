import numpy as np
import scipy.sparse.linalg as la
import matplotlib.pyplot as plt

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

		#u,s,vt = la.svds(self.M0)
		#print('SVD: ')
		#print(u.shape)
		#print(s.shape)
		#print(vt.shape)
		#S = sparse.csc_matrix((s,(np.arange(len(s)),np.arange(len(s)))),shape=(len(s),len(s)),dtype=np.float64)
		#vs = vt.transpose() * S
		#ut = u.transpose()
		#print(vs.shape)
		#print(ut.shape)
		#self.M0inv_pseudo = np.matmul(vs,ut)
		#print(self.M0inv_pseudo.shape)

	def diagnose_F(self,u1,u2,h1,h2):
		#Mu1 = PtU_u(self.topo,self.quad,self.dX,u1).M.transpose()
		#Mu2 = PtU_u(self.topo,self.quad,self.dX,u2).M.transpose()
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
		#Mu1 = PtU_u(self.topo,self.quad,self.dX,u1).M
		#Mu2 = PtU_u(self.topo,self.quad,self.dX,u2).M
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

	def plot_eig_vals(self,u):
		uq = plot_u(u,self.X,self.topo,self.topo_q,self.dX)
		Mu = PtU_u(self.topo,self.quad,self.dX,uq).M.transpose()
		L = self.M1 + 0.5*self.dt*self.M1*self.E10*self.M0inv*Mu
		R = self.M1 - 0.5*self.dt*self.M1*self.E10*self.M0inv*Mu
		Linv = la.inv(L)
		A = Linv*R
		w,v = la.eigs(A,k=len(u)-2)
		xx=np.linspace(0.0,1.0,1000000)
		plt.plot(xx,np.sqrt(1.0-xx*xx),c='k')
		plt.plot(xx,-np.sqrt(1.0-xx*xx),c='k')
		plt.xlim([1.0-0.00003,1.0+0.00001])
		plt.ylim([-0.003,0.003])
		plt.plot(w.real,w.imag,'o')
		plt.show()

	def solve(self,u1,h1):
		it = 0
		u2 = np.zeros(self.topo.nx*self.topo.n)
		h2 = np.zeros(self.topo.nx*self.topo.n)
		u2[:] = u1[:]
		h2[:] = h1[:]
		# begin newton iteration
		done = False;
		while not done:
			Fu = self.residual_u(u1,u2,h1,h2)
			Fh = self.residual_h(u1,u2,h1,h2)
			rhs = self.DM0inv*Fu - Fh
			dh = self.HELMinv*rhs
			du = -1.0*(self.M0invG*dh + self.M0inv*Fu)
			h2 = h2 + dh
			u2 = u2 + du
			# check the error
			dusq = np.dot(du,du)
			usq = np.dot(u2,u2)
			err_u = np.sqrt(dusq/usq)
			dhsq = np.dot(dh,dh)
			hsq = np.dot(h2,h2)
			err_h = np.sqrt(dhsq/hsq)
			print(str(it) + '\t|du|/|u|: ' + str(err_u) + '\t|dh|/|h|: ' + str(err_h))
			it = it + 1
			if err_u < 1.0e-12 and err_h < 1.0e-12:
				done = True

		u1[:] = u2[:]
		h1[:] = h2[:]

		return

	def solve_rk2(self,u1,h1):
		F1 = self.diagnose_F(u1,u1,h1,h1)
		Phi1 = self.diagnose_Phi(u1,u1,h1,h1)

		uh = u1 - self.dt*self.M0inv*self.E01*self.M1*Phi1
		hh = h1 - self.dt*self.E10*F1
		
		F2 = self.diagnose_F(u1,uh,h1,hh)
		Phi2 = self.diagnose_Phi(u1,uh,h1,hh)

		u2 = u1 - self.dt*self.M0inv*self.E01*self.M1*Phi2
		h2 = h1 - self.dt*self.E10*F2

		u1[:] = u2[:]
		h1[:] = h2[:]

		return

	def solve_imex(self,u1,h1):
		u2 = np.zeros(self.topo.nx*self.topo.n)
		up = np.zeros(self.topo.nx*self.topo.n)
		h2 = np.zeros(self.topo.nx*self.topo.n)

		# compute the provisional velocity
		Phi1 = self.diagnose_Phi(u1,u1,h1,h1)
		up = u1 - self.dt*self.grav*self.M0inv*self.E01*self.M1*Phi1

		# continuity rhs
		u1q = plot_u(u1,self.X,self.topo,self.topo_q,self.dX)
		upq = plot_u(up,self.X,self.topo,self.topo_q,self.dX)
		Mu1 = PtU_u(self.topo,self.quad,self.dX,u1q).M.transpose()
		Mup = PtU_u(self.topo,self.quad,self.dX,upq).M.transpose()
		u1h1 = Mu1*h1
		uph1 = Mup*h1
		F1 = (1.0/6.0)*self.M0inv*(2.0*u1h1 + uph1)
		rhs = self.M1*(h1 - self.dt*self.E10*F1)
		# continuity lhs
		Mu = PtU_u(self.topo,self.quad,self.dX,(1.0/6.0)*(2.0*upq + u1q)).M.transpose()
		A = self.M1 + self.dt*self.M1*self.E10*self.M0inv*Mu
		Ainv = la.inv(A)
		h2 = Ainv*rhs
		#A1 = self.dt*self.M1*self.E10
		#A2 = A1*self.M0inv_pseudo
		#A = self.M1 + A2*Mu
		#Ainv = np.linalg.inv(A)
		#h2 = np.matmul(Ainv,rhs).transpose()
		#print(h1.shape)
		#print(h2.shape)
		#print(Ainv.shape)

		# compute the final velocity
		Phi2 = self.diagnose_Phi(u1,up,h1,h2)
		u2 = u1 - self.dt*self.grav*self.M0inv*self.E01*self.M1*Phi2

		# compute the constraint
		#F2 = self.diagnose_F(u1,up,h1,h2);
		#uc = self.M0*F2
		#Mu1 = PtU_u(self.topo,self.quad,self.dX,u1q).M.transpose()
		#Mup = PtU_u(self.topo,self.quad,self.dX,upq).M.transpose()
		#uc = (1.0/6.0)*(2.0*Mu1*h1 + Mu1*h2 + Mup*h1 + 2.0*Mup*h2)
		#uo = np.zeros(self.topo.nx*self.topo.n)
		#J = self.dt*self.grav*self.E01*(Mu.transpose())

		#u2 = self.gmres(self.M0,up,self.M0*u1-self.dt*self.grav*self.E01*self.M1*Phi2)
		#u2 = self.gmres_constrained(self.M0+J,up,self.M0*u1-self.dt*self.grav*self.E01*self.M1*Phi2+J*up,uc)
		#du = self.gmres_constrained(self.M0+J,uo,self.M0*(u1-up)-self.dt*self.grav*self.E01*self.M1*Phi2,uc)
		#du = self.gmres_constrained_2(self.M0+J,uo,self.M0*(u1-up)-self.dt*self.grav*self.E01*self.M1*Phi2,uc)
		#u2 = up + du

		u1[:] = u2[:]
		h1[:] = h2[:]

		return

	def solve_imex_2(self,u1,h1):
		u2 = np.zeros(self.topo.nx*self.topo.n)
		up = np.zeros(self.topo.nx*self.topo.n)
		h2 = np.zeros(self.topo.nx*self.topo.n)

		# compute the provisional velocity
		Phi1 = self.diagnose_Phi(u1,u1,h1,h1)
		up = u1 - self.dt*self.grav*self.M0inv*self.E01*self.M1*Phi1

		# continuity equation
		u1q = plot_u(u1,self.X,self.topo,self.topo_q,self.dX)
		upq = plot_u(up,self.X,self.topo,self.topo_q,self.dX)
		Mu1 = PtU_u(self.topo,self.quad,self.dX,u1q).M.transpose()
		Mup = PtU_u(self.topo,self.quad,self.dX,upq).M.transpose()

		Fi = (1.0/6.0)*self.M0inv*(2.0*Mu1 + Mup)*h1
		A = self.M0 + (self.dt/6.0)*(Mu1 + 2.0*Mup)*self.E10
		Ainv = la.inv(A)
		rhs = Mu1*h1 + 2.0*Mup*h1 - self.dt*(Mu1*self.E10*Fi + 2.0*Mup*self.E10*Fi)
		Fj = (1.0/6.0)*Ainv*rhs

		F2 = Fi + Fj
		h2 = h1 - self.dt*self.E10*F2

		Phi2 = self.diagnose_Phi(u1,up,h1,h2)
		u2 = u1 - self.dt*self.grav*self.M0inv*self.E01*self.M1*Phi2

		u1[:] = u2[:]
		h1[:] = h2[:]

		return

	def gmres(self,A,x,b):
		max_size = 20
		H = np.zeros((2,1))
		Q = np.zeros((max_size,len(x)))

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
		return x + np.matmul(Q[:kk,:].transpose(),y)

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
