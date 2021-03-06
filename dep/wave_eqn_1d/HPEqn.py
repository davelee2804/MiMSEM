import numpy as np
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from Mats1D import *
from Assembly import *
from AssemblyNL import *
from BoundaryMat import *
from Proj import *

class HPEqn:
	def __init__(self,topo,quad,topo_q,lx,R,cp,eta_f,eta_h,Aeta,Beta,po,ps):
		self.topo = topo
		self.quad = quad
		self.topo_q = topo_q # topology for the quadrature points as 0 forms
		self.edge = LagrangeEdge(topo.n)
		self.R  = R
		self.cp = cp
		self.eta_f = eta_f 	# mass based vertical height at full levels
		self.eta_h = eta_h	# mass based vertical height at half levels
		self.Aeta = Aeta	# 
		self.Beta = Beta
		det = 0.5*lx/topo.nx
		self.detInv = 1.0/det
		self.po = po		# reference pressure (constant)
		self.ps = ps 		# surface pressure

		# layer thickness with respect to the generalised vertical coordiante, eta
		self.dEta = np.zeros((self.eta_h.shape[0]),dtype=np.float64)
		for k in np.arange(self.eta_h.shape[0]):
			self.dEta[k] = self.eta_f[k+1] - self.eta_f[k];

		# pseudo-density (vertical pressure gradient) within the layer
		self.pi = np.zeros((self.eta_h.shape[0],topo.nx*topo.n),dtype=np.float64)

		# 1 form matrix inverse
		self.M1 = Pmat(topo,quad).M
		self.M1inv = la.inv(self.M1)

		# 0 form matrix inverse
		self.M0 = Umat(topo,quad).M
		self.M0inv = la.inv(self.M0)

		# 1 form gradient matrix
		self.D10 = BoundaryMat(topo).M
		self.D01 = self.D10.transpose()

		# 0 form to 1 from gradient operator
		#self.A = self.detInv*self.D10

		# 1 form to 0 form gradient operator
		#self.B = -1.0*self.detInv*self.M0inv*self.D01*self.M1
		self.B = -1.0*self.M0inv*self.D01*self.M1

	# diagnose the layer density as the vertical derivative of the pressure
	def diag_pi(eta, pres):
		pi = np.zeros((pres.shape[0]-1,pres.shape[1]),dtype=np.float64)
		for k in np.arange(pres.shape[0] - 1):
			for i in np.arange(pres.shape[1]):
				pi[k,i] = (pres[k+1,i] - pres[k,i])/self.dEta[k]

		return pi

	# diagnose the mass flux field in each layer
	def diag_F(pi, vel):
		F = np.zeros(vel.shape,dtype=np.float64)
		for k in np.arange(pi.shape[0]):
			A = U_pi_mat(self.topo,self.quad,self.edge,pi[k,:]).M
			Af = np.dot(A,vel[k,:])
			F[k,:] = np.dot(self.M0inv,Af)

		return F

	# diagnose the pressure via the surface pressure
	def diag_p(pi, vel, pres):
		F = self.diag_F(pi, vel)

		# compute the surface pressure time derivative
		dps = np.zeros((pi.shape[1]),dtype=np.float64)
		for k in np.arange(pi.shape[0]):
			dFk = self.D10*F[k,:]
			for i in np.arange(pi.shape[1]):
				dps[i] = dps[i] + dFk[i]*self.dEta[k]

		# update the surface pressure
		self.ps = self.ps + dps

		# compute the pressure (skipping over the top level where
		# a rigid lid has been imposed)
		# TODO: does this need to be done in the weak form?
		for k = np.arange(pres.shape[0] - 1) + 1
			pres[k,:] = self.Aeta[k]*self.po + self.Beta[k]*ps[:]

	# diagnose the mass weighted specific volume, R.T.\pi/p
	def diag_C(tau, pres):
		C = np.zeros(tau.shape,dtype=np.float64)
		for k in np.arange(tau.shape[0]):
			Pk = P_pres_mat(self.topo,self.quad,self.dEta[k],pres[k,:],pres[k+1,:],False).M
			Pkinv = la.inv(Pk)
			M1tau = self.R*self.M1*tau[k,:]
			C[k,:] = Pkinv*M1tau

		return C

	# diagnose the geopotential at the layer mid levels
	def diag_Phi(C):
		Phi = np.zeros(C.shape,dtype=np.float64)
		Ck = np.zeros(C.shape[1],dtype=np.float64)
		for k in np.arange(C.shape[0]):
			Ck[:] = 0.0
			for j in np.arange(k-1):
				Ck[:] = Ck[:] + self.dEta[j]*C[j,:]

			Ck[:] = Ck[:] + 0.5*self.dEta[k]*C[k,:]
			# TODO: check that this relation holds pointwise
			Phi[k,:] = Ck[:]

		return Phi

	# diagnose the pressure gradient at the full levels
	def diag_pres_grad(pres):
		dp = np.zeros((pres.shape[0]-1,pres.shape[1]),dtype=np.float64)
		p_full = np.zeros((pres.shape[1]),dtype=np.float64)

		for k in np.arange(dp.shape[0]):
			# interpolate pressure to full levels
			p_full[:] = 0.5*(pres[k-1,:] + pres[k,:])
			Pp = self.M1*p_full
			dpk = -1.0*self.D01*Pp
			dp[k,:] = self.M0inv*dpk

		return dp

	# diagnose the mass weighted vertical velocity
	# this is done in the strong form (no galerkin projection)
	def diag_piw(F):
		piw = np.zeros((F.shape[0]+1,ps.shape[0]),dtype=np.float64)

		# assume 0 vertical velocity on top and bottom boundaries, 
		# so only evaluate internal values
		for k in np.arange(piw.shape[0]-2)+1:
			dF = self.D10*F[k,:]
			piw[k,:] = piw[k-1,:] - (self.Beta[k] - self.Beta[k-1])*self.ps[:] - dF[:]*self.dEta[k-1]

		return piw

	# diagnose the horiztonal kinetic energy
	def diag_ke(vel):
		ke = np.zeros((vel.shape[0],ps.shape[0]),dtype=np.float64)
		for k in np.arange(vel.shape[0]):
			K = U_vel_mat(self.topo,self.quad,vel[k,:]).M
			Ku = K*vel[k,:]
			ke[k,:] = 0.5*self.M1inv*Ku

		return ke

	# diagnose the vertical transport of horiztonal momentum
	# (via balance of kinetic and potential energy exchanges)
	def diag_G(ke,F,piw):
		G = np.zeros(F.shape,dtype=np.float64)
		for k in np.arange(F.shape[0]):
			Mf = U_vel_mat(self.topo,self.quad,F[k,:]).M
			Mfinv = la.inv(Mf)

			Mk = M01_pres_mat(self.topo,self.quad,ke).M
			dw = (piw[k+1,:] - piw[k,:])/self.dEta[k]
			Mkw = Mk*dw
			G[k,:] = -Mfinv*Mkw

		return G

	# diagnose the specific volume for compatibility with respect to the
	# transfer of kinetic to internal energy (as a 0 form)
	# TODO
	def diag_A(pres,vel,F,tau):
		A = np.zeros()

		return A
