import numpy as np

class Topo:
	def __init__(self,nx,n):
		self.nx = nx # number of elements
		self.n = n # polynomial order
		self.inds0 = np.zeros((n+1),dtype=np.int32)
		self.inds1 = np.zeros(n,dtype=np.int32)

	def localToGlobal1(self,el_x):
		for ix in np.arange(self.n):
			self.inds1[ix] = el_x*self.n + ix

		return self.inds1

	# map element indices to global matrix for the 0-forms
	#	assume periodic bcs
	def localToGlobal0(self,el_x):
		np1 = self.n+1
		for ix in np.arange(np1):
			if el_x == self.nx - 1 and ix == self.n:
				self.inds0[ix] = 0
			else:
				self.inds0[ix] = el_x*self.n + ix

		return self.inds0
