import numpy as np
from scipy import sparse

from Topo import *

# 0 to 1 form boundary matrix
class BoundaryMat:
	def __init__(self,topo):
		nfx = topo.n*topo.nx
		rows = np.zeros(2*nfx, dtype=np.int32)
		cols = np.zeros(2*nfx, dtype=np.int32)
		vals = np.zeros(2*nfx, dtype=np.int8)
		# row is the 1 form, column is the 0 form
		ii = 0
		# x-normal 1 forms 
		for f_x in np.arange(nfx):
			rows[ii+0] = f_x
			cols[ii+0] = f_x
			vals[ii+0] = -1

			rows[ii+1] = f_x
			cols[ii+1] = (f_x+1)%nfx
			vals[ii+1] = +1
			
			ii = ii + 2

		nr = nfx
		nc = nfx
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.int8)
