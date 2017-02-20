import numpy as np
from scipy import sparse

from Topo import *

# Transpose of the discrete curl operator
# see Hiemstra, Toshniwal, Huijmans and Gerritsma (2014) JCP Ex. 4.3 
class BoundaryMat:
	def __init__(self, topo):
		nfx = topo.n*topo.nx
		nfy = topo.n*topo.ny
		n2form = nfx*nfy
		rows = np.zeros(4*n2form, dtype=np.int32)
		cols = np.zeros(4*n2form, dtype=np.int32)
		vals = np.zeros(4*n2form, dtype=np.int8)
		shift = n2form
		ii = 0
		# row is the 2-form, column is the 1-form
		for f_y in np.arange(nfy):
			for f_x in np.arange(nfx):
				rows[ii+0] = f_y*nfx + f_x
				cols[ii+0] = f_y*nfx + f_x
				vals[ii+0] = -1

				rows[ii+1] = f_y*nfx + f_x
				if f_x == nfx - 1:
					cols[ii+1] = f_y*nfx
				else:
					cols[ii+1] = f_y*nfx + f_x + 1
				vals[ii+1] = +1

				rows[ii+2] = f_y*nfx + f_x
				cols[ii+2] = shift + f_y*nfx + f_x
				vals[ii+2] = +1

				rows[ii+3] = f_y*nfx + f_x
				if f_y == nfy - 1:
					cols[ii+3] = shift + f_x
				else:
					cols[ii+3] = shift + (f_y+1)*nfx + f_x
				vals[ii+3] = -1
				ii = ii + 4
				
		nr = nfx*nfy
		nc = 2*nfx*nfy
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.int8)

class BoundaryMatX:
	def __init__(self, topo):
		nfx = topo.n*topo.nx
		nfy = topo.n*topo.ny
		n2form = nfx*nfy
		rows = np.zeros(2*n2form, dtype=np.int32)
		cols = np.zeros(2*n2form, dtype=np.int32)
		vals = np.zeros(2*n2form, dtype=np.int8)
		ii = 0
		# row is the 2-form, column is the 1-form
		for f_y in np.arange(nfy):
			for f_x in np.arange(nfx):
				rows[ii+0] = f_y*nfx + f_x
				cols[ii+0] = f_y*nfx + f_x
				vals[ii+0] = -1

				rows[ii+1] = f_y*nfx + f_x
				if f_x == nfx - 1:
					cols[ii+1] = f_y*nfx
				else:
					cols[ii+1] = f_y*nfx + f_x + 1
				vals[ii+1] = +1

				ii = ii + 2
				
		nr = nfx*nfy
		nc = nfx*nfy
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.int8)

class BoundaryMatY:
	def __init__(self, topo):
		nfx = topo.n*topo.nx
		nfy = topo.n*topo.ny
		n2form = nfx*nfy
		rows = np.zeros(2*n2form, dtype=np.int32)
		cols = np.zeros(2*n2form, dtype=np.int32)
		vals = np.zeros(2*n2form, dtype=np.int8)
		ii = 0
		# row is the 2-form, column is the 1-form
		for f_y in np.arange(nfy):
			for f_x in np.arange(nfx):
				rows[ii] = f_y*nfx + f_x
				cols[ii] = f_y*nfx + f_x
				vals[ii] = +1

				rows[ii+1] = f_y*nfx + f_x
				if f_y == nfy - 1:
					cols[ii+1] = f_x
				else:
					cols[ii+1] = (f_y+1)*nfx + f_x
				vals[ii+1] = -1
				ii = ii + 2
				
		nr = nfx*nfy
		nc = nfx*nfy
		self.M = sparse.csc_matrix((vals,(rows,cols)),shape=(nr,nc),dtype=np.int8)
