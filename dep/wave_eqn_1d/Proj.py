import numpy as np
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from Mats1D import *
from Assembly import *

# Map spatial data to 1 forms
class Xto1:
	def __init__(self,topo,quad):
		M1 = Pmat(topo,quad).M
		M1inv = la.inv(M1)
		PtQ = PtQmat(topo,quad).M
		self.M = M1inv*PtQ

# Map spatial data to 0 forms
class Xto0:
	def __init__(self,topo,quad):
		M0 = Umat(topo,quad).M
		M0inv = la.inv(M0)
		UtQ = UtQmat(topo,quad).M
		self.M = M0inv*UtQ
