import numpy as np
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from Mats2D import *
from Assembly import *

# Map spatial data to 1 forms
class Xto1:
	def __init__(self,topo,quad):
		M1 = Umat(topo,quad).M
		M1inv = la.inv(M1)
		UtQ = UtQmat(topo,quad).M
		self.M = M1inv*UtQ

# Map spatial data to 2 forms
class Xto2:
	def __init__(self,topo,quad):
		M2 = Wmat(topo,quad).M
		M2inv = la.inv(M2)
		WtQ = WtQmat(topo,quad).M
		self.M = M2inv*WtQ

# Map spatial data to 0 forms
class Xto0:
	def __init__(self,topo,quad):
		M0 = Pmat(topo,quad).M
		M0inv = la.inv(M0)
		PtQ = PtQmat(topo,quad).M
		self.M = M0inv*PtQ

# Map the tangent velocities to normal velocities
class UNormToTang:
	def __init__(self,topo,quad):
		self.shift = topo.nx*topo.ny*topo.n*topo.n

		Uxx = UxxMat(topo,quad).M
		Uxy = UxyMat(topo,quad).M
		Uyx = UyxMat(topo,quad).M
		Uyy = UyyMat(topo,quad).M

		UxxInv = la.inv(Uxx)
		UyyInv = la.inv(Uyy)

		# U tangent to normal transformation matrices
		self.Uxtn = UxxInv*Uxy
		self.Uytn = UyyInv*Uyx

	def apply(self,un):
		# Normal velocities
		uxn = un[:self.shift]
		uyn = un[self.shift:]

		# Tangent velocities
		uxt = self.Uxtn*uyn
		uyt = self.Uytn*uxn

		ut = np.zeros((2*self.shift),dtype=np.float64)
		ut[:self.shift] = uxt
		ut[self.shift:] = uyt

		return ut

