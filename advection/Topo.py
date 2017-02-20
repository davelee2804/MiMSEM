import numpy as np

class Topo:
	def __init__(self,nx,ny,n):
		self.nx = nx # number of elements in x dimension
		self.ny = ny # number of elements in y dimension
		self.n = n # polynomial order

	# map element indices to global matrix for the 2-forms
	def localToGlobal2(self,el_x,el_y):
		n2 = self.n*self.n
		ginds = np.zeros(n2,dtype=np.int32)
		for iy in np.arange(self.n):
			for ix in np.arange(self.n):
				ginds[iy*self.n+ix] = el_y*self.nx*n2 + iy*self.nx*self.n + el_x*self.n + ix

		return ginds

	# map element indices to global matrix for the 1-forms
	#	assume periodic in both dimensions
	#	x-normal first then y-normal
	def localToGlobal1(self,el_x,el_y):
		n2 = self.n*self.n
		np1 = self.n+1
		ginds = np.zeros(2*np1*self.n,dtype=np.int32)
		shift = (self.n*self.nx)*(self.n*self.ny)
		for iy in np.arange(self.n):
			for ix in np.arange(np1):
				if el_x == self.nx - 1 and ix == self.n:
					ginds[iy*np1+ix] = el_y*self.nx*n2 + iy*self.nx*self.n
				else:
					ginds[iy*np1+ix] = el_y*self.nx*n2 + iy*self.nx*self.n + el_x*self.n + ix

		for iy in np.arange(np1):
			for ix in np.arange(self.n):
				if el_y == self.ny - 1 and iy == self.n:
					ginds[np1*self.n+iy*self.n+ix] = shift + el_x*self.n + ix
				else:
					ginds[np1*self.n+iy*self.n+ix] = shift + el_y*self.nx*n2 + iy*self.nx*self.n + el_x*self.n + ix

		return ginds

	def localToGlobal1x(self,el_x,el_y):
		n2 = self.n*self.n
		np1 = self.n+1
		ginds = np.zeros(np1*self.n,dtype=np.int32)
		for iy in np.arange(self.n):
			for ix in np.arange(np1):
				if el_x == self.nx - 1 and ix == self.n:
					ginds[iy*np1+ix] = el_y*self.nx*n2 + iy*self.nx*self.n
				else:
					ginds[iy*np1+ix] = el_y*self.nx*n2 + iy*self.nx*self.n + el_x*self.n + ix

		return ginds

	def localToGlobal1y(self,el_x,el_y):
		n2 = self.n*self.n
		np1 = self.n+1
		ginds = np.zeros(np1*self.n,dtype=np.int32)

		for iy in np.arange(np1):
			for ix in np.arange(self.n):
				if el_y == self.ny - 1 and iy == self.n:
					ginds[iy*self.n+ix] = el_x*self.n + ix
				else:
					ginds[iy*self.n+ix] = el_y*self.nx*n2 + iy*self.nx*self.n + el_x*self.n + ix

		return ginds

	# map element indices to global matrix for the 0-forms
	#	assume periodic in both dimensions
	def localToGlobal0(self,el_x,el_y):
		n2 = self.n*self.n
		np1 = self.n+1
		ginds = np.zeros(np1*np1,dtype=np.int32)
		for iy in np.arange(np1):
			for ix in np.arange(np1):
				if el_x == self.nx - 1 and ix == self.n and el_y == self.ny - 1 and iy == self.n:
					ginds[iy*np1+ix] = 0
				elif el_x == self.nx - 1 and ix == self.n:
					ginds[iy*np1+ix] = el_y*self.nx*n2 + iy*self.nx*self.n
				elif el_y == self.ny - 1 and iy == self.n:
					ginds[iy*np1+ix] = el_x*self.n + ix
				else:
					ginds[iy*np1+ix] = el_y*self.nx*n2 + iy*self.nx*self.n + el_x*self.n + ix

		return ginds
