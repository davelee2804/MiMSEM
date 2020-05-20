import numpy as np
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from Mats1D import *
from Assembly import *

# Map spatial data to 1 forms
class Xto1:
	def __init__(self,topo,quad,dX):
		M1 = Pmat(topo,quad,dX).M
		M1inv = la.inv(M1)
		PtQ = PtQmat(topo,quad,dX).M
		self.M = M1inv*PtQ

# Map spatial data to 0 forms
class Xto0:
	def __init__(self,topo,quad,dX):
		M0 = Umat(topo,quad,dX).M
		M0inv = la.inv(M0)
		UtQ = UtQmat(topo,quad,dX).M
		self.M = M0inv*UtQ

# Map spatial data to 1 forms
class Xto1_up:
	def __init__(self,topo,quad,dX):
		node = LagrangeNode_up(topo.n,quad.n)
		edge = LagrangeEdge_up(topo.n,quad.n)
		M1 = Pmat_up(topo,quad,dX,edge.M_ij_r,edge.M_ij_r).M
		M1inv = la.inv(M1)
		PtQ = PtQmat_up(topo,quad,dX,edge.M_ij_r).M
		self.M = M1inv*PtQ

# Map spatial data to 0 forms
class Xto0_up:
	def __init__(self,topo,quad,dX):
		node = LagrangeNode_up(topo.n,quad.n)
		edge = LagrangeEdge_up(topo.n,quad.n)
		M0 = Umat_up(topo,quad,dX,node.M_ij_r,node.M_ij_r).M
		M0inv = la.inv(M0)
		UtQ = UtQmat_up(topo,quad,dX,node.M_ij_r).M
		self.M = M0inv*UtQ
