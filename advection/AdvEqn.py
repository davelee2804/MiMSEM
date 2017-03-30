import numpy as np
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from Mats2D import *
from Assembly import *
from BoundaryMat import *
from Proj import *
from LieDeriv import *

class AdvectionEqn:
	def __init__(self,topo,quad,lx,ly):
		n2forms = topo.nx*topo.ny*topo.n*topo.n

		self.qh = np.zeros((n2forms),dtype=np.float64)
		self.qf = np.zeros((n2forms),dtype=np.float64)

		self.lie = LieDeriv(topo,quad,lx,ly)

	def solveRK2(self,u,qi,dt,do_assembly):
		qh = self.qh
		qf = self.qf

		duqi = self.lie.assemble(u,qi,do_assembly)
		qh = qi + 0.5*dt*duqi
		duqh = self.lie.assemble(u,qh,do_assembly)
		qf = qi + dt*duqh

		return qf
		
