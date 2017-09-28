#!/usr/bin/env python

import numpy as np

# container class for the processor on a given side of the
# target processor and its relative coordinate system transformation
class Side:
	def __init__(self,proc,axis):
		self.proc = proc
		self.axis = axis

class Proc:
	def __init__(self,pn,nx,pi):
		self.pn = pn	# polynomial degree
		self.nx = nx	# number of elements in each dimension
		self.pi = pi	# global processor id

		self.pn1 = self.pn+1
		self.pnx = self.nx*self.pn
		self.pnx1 = self.nx*self.pn+1
		self.xshift = self.pnx1*self.pnx

		# adjacent processors
		self.SS = None
		self.SE = None
		self.EE = None
		self.EN = None
		self.NN = None
		self.NW = None
		self.WW = None
		self.WS = None

		# local index arrays
		self.inds0 = np.zeros((self.pn1)*(self.pn1),dtype=np.int32)
		self.inds1x = np.zeros((self.pn1)*(self.pn),dtype=np.int32)
		self.inds1y = np.zeros((self.pn)*(self.pn1),dtype=np.int32)
		self.inds2 = np.zeros((self.pn)*(self.pn),dtype=np.int32)

	# return the local indices for the nodes of given element
	def localToGlobal0(self,ex,ey):
		kk = 0
		for iy in np.arange(self.pn1):
			for ix in np.arange(self.pn1):
				inds0[kk] = (ey*self.pn+iy)*self.pnx1 + ex*self.pn + ix
				kk = kk + 1

		return inds0

	# return the local indices for the nodes of given element
	def localToGlobal1x(self,ex,ey):
		kk = 0
		for iy in np.arange(self.pn):
			for ix in np.arange(self.pn1):
				inds1x[kk] = (ey*self.pn+iy)*self.pnx1 + ex*self.pn + ix
				kk = kk + 1

		return inds1x

	# return the local indices for the nodes of given element
	def localToGlobal1y(self,ex,ey):
		kk = 0
		for iy in np.arange(self.pn1):
			for ix in np.arange(self.pn):
				inds1y[kk] = (ey*self.pn+iy)*self.pnx + ex*self.pn + ix
				kk = kk + 1

		return inds1y + self.xshift

	# return the local indices for the nodes of given element
	def localToGlobal2(self,ex,ey):
		kk = 0
		for iy in np.arange(self.pn):
			for ix in np.arange(self.pn):
				inds2[kk] = (ey*self.pn+iy)*self.pnx + ex*self.pn + ix
				kk = kk + 1

		return inds2

# face of the cube
class Face:
	def __init__(self,fi,npx):
		self.fi = fi	# face index
		self.npx = npx	# number of processors in x

		self.procs = [None]*(npx*npx)

		self.S = None
		self.E = None
		self.N = None
		self.W = None

		self.Saxis = np.zeros((2,2),dtype=np.int8)
		self.Eaxis = np.zeros((2,2),dtype=np.int8)
		self.Naxis = np.zeros((2,2),dtype=np.int8)
		self.Waxis = np.zeros((2,2),dtype=np.int8)

		self.sideProcs = [None]*npx

	# return the set of processors on any give side of the face
	def getS(self,orient):
		for pi in np.arange(self.npx):
			self.sideProcs[pi] = self.procs[pi]

		if orient == -1:
			self.sideProcs = self.sideProcs[::-1]

		return self.sideProcs

	def getE(self,orient):
		for pi in np.arange(self.npx):
			self.sideProcs[pi] = self.procs[pi*self.npx + self.npx - 1]

		if orient == -1:
			self.sideProcs = self.sideProcs[::-1]

		return self.sideProcs

	def getN(self,orient):
		for pi in np.arange(self.npx):
			self.sideProcs[pi] = self.procs[(self.npx)*(self.npx-1)+pi]

		if orient == -1:
			self.sideProcs = self.sideProcs[::-1]

		return self.sideProcs

	def getW(self,orient):
		for pi in np.arange(self.npx):
			self.sideProcs[pi] = self.procs[pi*self.npx]

		if orient == -1:
			self.sideProcs = self.sideProcs[::-1]

		return self.sideProcs

# topological cube of 6.n^2 processors
class ParaCube:
	def __init__(self,n_procs,pn,nx):
		self.np = n_procs	# total number of processors
		self.pn = pn		# polynomial degree
		self.nx = nx		# number of elements across a face

		npx = int(np.sqrt(n_procs/6))
		self.npx = npx
		print 'no. procs in each dimsion: ' + str(self.npx)

		self.faces = [None]*6
		for fi in np.arange(6):
			self.faces[fi] = Face(fi,npx)

		# set the face adjacencies
		self.faces[0].S = self.faces[4]
		self.faces[0].E = self.faces[2]
		self.faces[0].N = self.faces[1]
		self.faces[0].W = self.faces[5]

		self.faces[1].S = self.faces[0]
		self.faces[1].E = self.faces[2]
		self.faces[1].N = self.faces[3]
		self.faces[1].W = self.faces[5]

		self.faces[2].S = self.faces[0]
		self.faces[2].E = self.faces[4]
		self.faces[2].N = self.faces[3]
		self.faces[2].W = self.faces[1]

		self.faces[3].S = self.faces[2]
		self.faces[3].E = self.faces[4]
		self.faces[3].N = self.faces[5]
		self.faces[3].W = self.faces[1]

		self.faces[4].S = self.faces[2]
		self.faces[4].E = self.faces[0]
		self.faces[4].N = self.faces[5]
		self.faces[4].W = self.faces[3]

		self.faces[5].S = self.faces[4]
		self.faces[5].E = self.faces[0]
		self.faces[5].N = self.faces[1]
		self.faces[5].W = self.faces[3]

		# set the face orientations
		self.faces[0].Saxis[0][1] = -1; self.faces[0].Saxis[1][0] = +1
		self.faces[0].Eaxis[0][1] = +1; self.faces[0].Eaxis[1][0] = -1
		self.faces[0].Naxis[0][0] = +1; self.faces[0].Naxis[1][1] = +1
		self.faces[0].Waxis[0][0] = +1; self.faces[0].Waxis[1][1] = +1

		self.faces[1].Saxis[0][0] = +1; self.faces[1].Saxis[1][1] = +1
		self.faces[1].Eaxis[0][0] = +1; self.faces[1].Eaxis[1][1] = +1
		self.faces[1].Naxis[0][1] = -1; self.faces[1].Naxis[1][0] = +1
		self.faces[1].Waxis[0][1] = +1; self.faces[1].Waxis[1][0] = -1

		self.faces[2].Saxis[0][1] = -1; self.faces[2].Saxis[1][0] = +1
		self.faces[2].Eaxis[0][1] = +1; self.faces[2].Eaxis[1][0] = -1
		self.faces[2].Naxis[0][0] = +1; self.faces[2].Naxis[1][1] = +1
		self.faces[2].Waxis[0][0] = +1; self.faces[2].Waxis[1][1] = +1

		self.faces[3].Saxis[0][0] = +1; self.faces[3].Saxis[1][1] = +1##
		self.faces[3].Eaxis[0][0] = +1; self.faces[3].Eaxis[1][1] = +1
		self.faces[3].Naxis[0][1] = -1; self.faces[3].Naxis[1][0] = +1
		self.faces[3].Waxis[0][1] = +1; self.faces[3].Waxis[1][0] = -1

		self.faces[4].Saxis[0][1] = -1; self.faces[4].Saxis[1][0] = +1
		self.faces[4].Eaxis[0][1] = +1; self.faces[4].Eaxis[1][0] = -1
		self.faces[4].Naxis[0][0] = +1; self.faces[4].Naxis[1][1] = +1
		self.faces[4].Waxis[0][0] = +1; self.faces[4].Waxis[1][1] = +1

		self.faces[5].Saxis[0][0] = +1; self.faces[5].Saxis[1][1] = +1
		self.faces[5].Eaxis[0][0] = +1; self.faces[5].Eaxis[1][1] = +1
		self.faces[5].Naxis[0][1] = -1; self.faces[5].Naxis[1][0] = +1
		self.faces[5].Waxis[0][1] = +1; self.faces[5].Waxis[1][0] = -1

		self.procs = [None]*self.np

		# stitch together the connectivity of processors on a given cube face
		pi = 0
		for fi in np.arange(6):
			face = self.faces[fi]
			for pj in np.arange(npx*npx):
				self.procs[pi] = Proc(self.pn,self.nx,pi)
				face.procs[pj] = self.procs[pi]
				pi = pi + 1

			# set the neighbour processors on this face
			shift = fi*npx*npx
			pj = shift
			# all processors on the same face share the same orientation
			axis = np.zeros((2,2),dtype=np.int8)
			axis[0,0] = +1
			axis[1,1] = +1
			for py in np.arange(npx):
				for px in np.arange(npx):
					proc = self.procs[pj]
					if px < npx - 1:
						proc.EE = Side(self.procs[shift + py*npx + px + 1],axis)
					if px < npx - 1 and py < npx - 1:
						proc.EN = Side(self.procs[shift + (py+1)*npx + px + 1],axis)
					if py < npx - 1:
						proc.NN = Side(self.procs[shift + (py+1)*npx + px],axis)
					if px > 0 and py < npx - 1:
						proc.NW = Side(self.procs[shift + (py+1)*npx + px - 1],axis)
					if px > 0:
						proc.WW = Side(self.procs[shift + py*npx + px - 1],axis)
					if px > 0 and py > 0:
						proc.WS = Side(self.procs[shift + (py-1)*npx + px - 1],axis)
					if py > 0:
						proc.SS = Side(self.procs[shift + (py-1)*npx + px],axis)
					if px < npx - 1 and py > 0:
						proc.SE = Side(self.procs[shift + (py-1)*npx + px + 1],axis)

					pj = pj + 1

		# stitch together the connectivity of processors on neighbouring cube faces
		for fi in np.arange(6):
			face = self.faces[fi]

			# neighbour processors on the south edge
			south = face.S
			axis = face.Saxis
			if axis[1][1] == +1:
				sideProcs = south.getN(+1)
			elif axis[1][0] == +1:
				sideProcs = south.getE(-1)
			else:
				print 'adjacency error (1.0)'

			for pi in np.arange(npx):
				proc = face.procs[pi]
				proc.SS = Side(sideProcs[pi],axis)
				if pi > 0:
					proc.WS = Side(sideProcs[pi-1],axis)
				if pi < npx - 1:
					proc.SE = Side(sideProcs[pi+1],axis)

			# neighbour processors on the east edge
			east = face.E
			axis = face.Eaxis
			if axis[0][0] == +1:
				sideProcs = east.getW(+1)
			elif axis[0][1] == +1:
				sideProcs = east.getS(-1)
			else:
				print 'adjacency error (1.1)'

			for pi in np.arange(npx):
				proc = face.procs[pi*npx + npx - 1]
				proc.EE = Side(sideProcs[pi],axis)
				if pi > 0:
					proc.SE = Side(sideProcs[pi-1],axis)
				if pi < npx - 1:
					proc.EN = Side(sideProcs[pi+1],axis)

			# neighbour processors on the north edge
			north = face.N
			axis = face.Naxis
			if axis[1][1] == +1:
				sideProcs = north.getS(+1)
			elif axis[1][0] == +1:
				sideProcs = north.getW(-1)
			else:
				print 'adjacency error (1.2)'

			for pi in np.arange(npx):
				proc = face.procs[npx*(npx-1) + pi]
				proc.NN = Side(sideProcs[pi],axis)
				if pi > 0:
					proc.NW = Side(sideProcs[pi-1],axis)
				if pi < npx - 1:
					proc.EN = Side(sideProcs[pi+1],axis)

			# nieghbour processors on the west edge
			west = face.W
			axis = face.Waxis
			if axis[0][0] == +1:
				sideProcs = west.getE(+1)
			elif axis[0][1] == +1:
				sideProcs = west.getN(-1)
			else:
				print 'adjacency error (1.3)'

			for pi in np.arange(npx):
				proc = face.procs[npx*pi]
				proc.WW = Side(sideProcs[pi],axis)
				if pi > 0:
					proc.WS = Side(sideProcs[pi-1],axis)
				if pi < npx - 1:
					proc.NW = Side(sideProcs[pi+1],axis)

	def print_procs(self):
		neighbours = np.zeros(8,dtype=np.int32)

		for pi in np.arange(self.np):
			proc = self.procs[pi]
			neighbours[0] = proc.SS.proc.pi
			if proc.SE == None:
				neighbours[1] = -1
			else:
				neighbours[1] = proc.SE.proc.pi
			neighbours[2] = proc.EE.proc.pi
			if proc.EN == None:
				neighbours[3] = -1
			else:
				neighbours[3] = proc.EN.proc.pi
			neighbours[4] = proc.NN.proc.pi
			if proc.NW == None:
				neighbours[5] = -1
			else:
				neighbours[5] = proc.NW.proc.pi
			neighbours[6] = proc.WW.proc.pi
			if proc.WS == None:
				neighbours[7] = -1
			else:
				neighbours[7] = proc.WS.proc.pi

			print str(pi) + ':\t' + str(neighbours)

	def print_neighbours(self,pi):
		neighbours = np.zeros(8,dtype=np.int32)
		proc = self.procs[pi]

		neighbours[0] = proc.SS.proc.pi
		if proc.SE == None:
			neighbours[1] = -1
		else:
			neighbours[1] = proc.SE.proc.pi
		neighbours[2] = proc.EE.proc.pi
		if proc.EN == None:
			neighbours[3] = -1
		else:
			neighbours[3] = proc.EN.proc.pi
		neighbours[4] = proc.NN.proc.pi
		if proc.NW == None:
			neighbours[5] = -1
		else:
			neighbours[5] = proc.NW.proc.pi
		neighbours[6] = proc.WW.proc.pi
		if proc.WS == None:
			neighbours[7] = -1
		else:
			neighbours[7] = proc.WS.proc.pi

		print str(pi) + ':\t' + str(neighbours)

pc = ParaCube(24,3,4)

print 'results for 24 procs'
print '0:\t[19 17  1  3  2 23 21 -1'
pc.print_neighbours(0)
print '1:\t[17 -1  9  8  3  2  0 19'
pc.print_neighbours(1)
print '2:\t[ 0  1  3  5  4 -1 23 21'
pc.print_neighbours(2)
print '3:\t[ 1  9  8 -1  5  4  2  0'
pc.print_neighbours(3)
print '4:\t[ 2  3  5  7  6 22 23 -1'
pc.print_neighbours(4)
print '5:\t[ 3 -1  8 10  7  6  4  2'
pc.print_neighbours(5)
print '6:\t[ 4  5  7 12 14 -1 22 23'
pc.print_neighbours(6)
print '7:\t[ 5  8 10 -1 12 14  6  4'
pc.print_neighbours(7)
print '8:\t[ 3  1  9 11 10  7  5 -1'
pc.print_neighbours(8)
print '9:\t[ 1 -1 17 16 11 10  8  3'
pc.print_neighbours(9)
print '10:\t[ 8  9 11 13 12 -1  7  5'
pc.print_neighbours(10)
print '11:\t[ 9 17 16 -1 13 12 10  8'
pc.print_neighbours(11)
print '12:\t[10 11 13 15 14  6  7 -1'
pc.print_neighbours(12)
print '13:\t[11 -1 16 18 15 14 12 10'
pc.print_neighbours(13)
print '14:\t[12 13 15 20 22 -1  6  7'
pc.print_neighbours(14)
print '15:\t[13 16 18 -1 20 22 14 12'
pc.print_neighbours(15)
print '16:\t[11  9 17 19 18 15 13 -1'
pc.print_neighbours(16)
print '17:\t[ 9 -1  1  0 19 18 16 11'
pc.print_neighbours(17)
print '18:\t[16 17 19 21 20 -1 15 13'
pc.print_neighbours(18)
print '19:\t[17  1  0 -1 21 20 18 16'
pc.print_neighbours(19)
print '20:\t[18 19 21 23 22 14 15 -1'
pc.print_neighbours(20)
print '21:\t[19 -1  0  2 23 22 20 18'
pc.print_neighbours(21)
print '22:\t[20 21 24  4  6 -1 14 15'
pc.print_neighbours(22)
print '23:\t[21  0  2 -1  4  6 22 20'
pc.print_neighbours(23)

#pc.print_procs()
