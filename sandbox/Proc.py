#!/usr/bin/env python

import numpy as np

# container class for the processor on a given side of the
# target processor and its relative coordinate system transformation
class Side:
	def __init__(self,proc,axis):
		self.proc = proc
		self.axis = axis

class Proc:
	def __init__(self,pn,nx,pi,fi,n_procs):
		self.pn = pn	# polynomial degree
		self.nx = nx	# number of elements in each dimension (local)
		self.pi = pi	# global processor id
		self.fi = fi	# index of face that owns this processor
		self.np = n_procs # global number of processors

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

		# global index arrays for sides of processor region
		self.side0 = np.zeros(self.nx*self.pn+1,dtype=np.int32)
		self.side1 = np.zeros(self.nx*self.pn,dtype=np.int32)

	# define global indices for local nodes/edges/faces
	def buildLocalArrays(self):
		# number of local nodes/edges/faces
		self.n0lavg = (self.nx*self.pn)*(self.nx*self.pn)
		self.n0l = self.n0lavg
		# account for the two hanging nodes on the cube
		# note: these two hanging nodes have the last two global indices
		if self.fi == 0 and self.SE == None:
			self.n0l = self.n0lavg + 1
		if self.fi == 1 and self.NW == None:
			self.n0l = self.n0lavg + 1
		self.n1xl = (self.nx*self.pn)*(self.nx*self.pn)
		self.n1yl = (self.nx*self.pn)*(self.nx*self.pn)
		self.n2l = (self.nx*self.pn)*(self.nx*self.pn)

		# number of global nodes/edges/faces (on this processor)
		self.n0g = (self.nx*self.pn+1)*(self.nx*self.pn+1)
		self.n1xg = (self.nx*self.pn+1)*(self.nx*self.pn)
		self.n1yg = (self.nx*self.pn)*(self.nx*self.pn+1)
		self.n2g = (self.nx*self.pn)*(self.nx*self.pn)

		# global offsets
		self.shift0 = self.pi*self.n0lavg
		self.shift1x = self.pi*self.n1xl
		self.shift1y = self.np*self.n1xl + self.pi*self.n1yl
		self.shift2 = self.pi*self.n2l

		# global index arrays
		self.loc0 = np.zeros(self.n0g,dtype=np.int32)
		self.loc1x = np.zeros(self.n1xg,dtype=np.int32)
		self.loc1y = np.zeros(self.n1yg,dtype=np.int32)
		self.loc2 = np.zeros(self.n2g,dtype=np.int32)

		# global indices of local nodes
		i_node = self.shift0
		for iy in np.arange(self.nx*self.pn):
			for ix in np.arange(self.nx*self.pn):
				self.loc0[iy*(self.nx*self.pn+1) + ix] = i_node
				i_node = i_node + 1

		# hanging nodes
		if self.fi == 0 and self.SE == None:
			self.loc0[self.nx*self.pn] = self.np*self.n0lavg
		if self.fi == 1 and self.NW == None:
			self.loc0[(self.nx*self.pn)*(self.nx*self.pn+1)] = self.np*self.n0lavg + 1

		# global indices of local x normal edges
		i_edge = self.shift1x
		for iy in np.arange(self.nx*self.pn):
			for ix in np.arange(self.nx*self.pn):
				self.loc1x[iy*(self.nx*self.pn+1) + ix] = i_edge
				i_edge = i_edge + 1

		# global indices of local y normal edges
		i_edge = self.shift1y
		for iy in np.arange(self.nx*self.pn):
			for ix in np.arange(self.nx*self.pn):
				self.loc1y[iy*(self.nx*self.pn) + ix] = i_edge
				i_edge = i_edge + 1

		# global indices of local faces

	# define global indices for remote nodes/edges/faces
	def buildGlobalArrays(self):
		nxp = self.nx*self.pn
		nxp1 = self.nx*self.pn + 1

		east = self.EE.proc
		north = self.NN.proc
		eAxis = self.EE.axis
		nAxis = self.NN.axis
		if self.EN != None:
			northEast = self.EN.proc
		else:
			northEast = None
		if self.SE != None:
			southEast = self.SE.proc
		else:
			southEast = None
		if self.NW != None:
			northWest = self.NW.proc
		else:
			northWest = None

		# 1. east and north processor neighbours are on the same face
		if eAxis[0][0] == +1 and nAxis[1][1] == +1:
			gInds0 = east.getW0(+1)
			for iy in np.arange(nxp1):
				self.loc0[iy*nxp1 + nxp] = gInds0[iy]

			gInds0 = north.getS0(+1)
			for ix in np.arange(nxp1):
				self.loc0[nxp*nxp1 + ix] = gInds0[ix]

			gInds0 = northEast.getS0(+1)
			self.loc0[nxp1*nxp1 - 1] = gInds0[0]

		# 2. north proc on same axis and east proc rotated -pi/2
		elif nAxis[1][1] == +1 and eAxis[0][1] == +1:
			gInds0 = north.getS0(+1)
			for ix in np.arange(nxp1):
				self.loc0[nxp*nxp1 + ix] = gInds0[ix]

			gInds0 = east.getS0(-1)
			for iy in np.arange(nxp1):
				self.loc0[iy*nxp1 + nxp] = gInds0[iy]

			if southEast == None:
				self.loc0[nxp] = self.np*self.n0lavg
			else:
				gInds0 = southEast.getS0(+1)
				self.loc0[nxp] = gInds0[0]
				
		# 3. east proc on same axis and noth proc rotated +pi/2
		elif eAxis[0][0] == +1 and nAxis[1][0] == +1:
			gInds0 = east.getW0(+1)
			for iy in np.arange(nxp1):
				self.loc0[iy*nxp1 + nxp] = gInds0[iy]

			gInds0 = north.getW0(-1)
			for ix in np.arange(nxp1):
				self.loc0[nxp*nxp1 + ix] = gInds0[ix]

			if northWest == None:
				self.loc0[nxp*nxp1] = self.np*self.n0lavg + 1
			else:
				gInds0 = northWest.getS0(+1)
				self.loc0[nxp*nxp1] = gInds0[0]

		else:
			print 'adjacency error (2.0)'

		# now do the edges (north and south procs on same face)
		if eAxis[0][0] == +1 and nAxis[1][1] == +1:
			gInds1x = east.getW1(+1)
			for iy in np.arange(nxp):
				self.loc1x[iy*nxp1 + nxp] = gInds1x[iy]

			gInds1y = north.getS1(+1)
			for ix in np.arange(nxp):
				self.loc1y[nxp*nxp + ix] = gInds1y[ix]

		# 2. north proc on same axis and east proc rotated -pi/2
		elif nAxis[1][1] == +1 and eAxis[0][1] == +1:
			gInds1y = north.getS1(+1)
			for ix in np.arange(nxp):
				self.loc1y[nxp*nxp + ix] = gInds1y[ix]

			gInds1x = east.getS1(-1)
			for iy in np.arange(nxp):
				self.loc1x[iy*nxp1 + nxp] = gInds1x[iy]

		# 3. east proc on same axis and noth proc rotated +pi/2
		elif eAxis[0][0] == +1 and nAxis[1][0] == +1:
			gInds1x = east.getW1(+1)
			for iy in np.arange(nxp):
				self.loc1x[iy*nxp1 + nxp] = gInds1x[iy]

			gInds1y = north.getW1(-1)
			for ix in np.arange(nxp):
				self.loc1y[nxp*nxp + ix] = gInds1y[ix]

		else:
			print 'adjacency error (2.1)'

	# return the local indices for the nodes of given element
	def elementToLocal0(self,ex,ey):
		kk = 0
		for iy in np.arange(self.pn1):
			for ix in np.arange(self.pn1):
				inds0[kk] = (ey*self.pn+iy)*self.pnx1 + ex*self.pn + ix
				kk = kk + 1

		return inds0

	# return the local indices for the x normal edges of given element
	def elementToLocal1x(self,ex,ey):
		kk = 0
		for iy in np.arange(self.pn):
			for ix in np.arange(self.pn1):
				inds1x[kk] = (ey*self.pn+iy)*self.pnx1 + ex*self.pn + ix
				kk = kk + 1

		return inds1x

	# return the local indices for the y normal edges of given element
	def elementToLocal1y(self,ex,ey):
		kk = 0
		for iy in np.arange(self.pn1):
			for ix in np.arange(self.pn):
				inds1y[kk] = (ey*self.pn+iy)*self.pnx + ex*self.pn + ix
				kk = kk + 1

		return inds1y + self.xshift

	# return the local indices for the faces of given element
	def elementToLocal2(self,ex,ey):
		kk = 0
		for iy in np.arange(self.pn):
			for ix in np.arange(self.pn):
				inds2[kk] = (ey*self.pn+iy)*self.pnx + ex*self.pn + ix
				kk = kk + 1

		return inds2

	# return the global node indices on the west side of the processor region
	def getW0(self,orient):
		nxp1 = self.nx*self.pn + 1
		for iy in np.arange(nxp1):
			self.side0[iy] = self.loc0[iy*nxp1]

		if orient == -1:
			self.side0[:] = self.side0[::-1]

		return self.side0

	# return the global edge indices on the west side of the processor region
	def getW1(self,orient):
		nxp1 = self.nx*self.pn+1
		nyp = self.nx*self.pn
		for iy in np.arange(nyp):
			self.side1[iy] = self.loc1x[iy*nxp1]

		if orient == -1:
			self.side1 = self.side1[::-1]

		return self.side1

	# return the global node indices on the south side of the processor region
	def getS0(self,orient):
		nxp1 = self.nx*self.pn + 1
		for ix in np.arange(nxp1):
			self.side0[ix] = self.loc0[ix]

		if orient == -1:
			self.side0[:] = self.side0[::-1]

		return self.side0

	# return the global edge indices on the south side of the processor region
	def getS1(self,orient):
		nxp = self.nx*self.pn
		for ix in np.arange(nxp):
			self.side1[ix] = self.loc1y[ix]

		if orient == -1:
			self.side1 = self.side1[::-1]

		return self.side1

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
		self.nx = nx		# number of elements across a face (global)

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

		self.faces[3].Saxis[0][0] = +1; self.faces[3].Saxis[1][1] = +1
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
				self.procs[pi] = Proc(self.pn,self.nx/self.npx,pi,fi,self.np)
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

		# build the global index maps for the nodes/edges/faces on a given processor
		for pi in np.arange(self.np):
			self.procs[pi].buildLocalArrays()
		for pi in np.arange(self.np):
			self.procs[pi].buildGlobalArrays()

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

	def print_nodes(self,pi):
		proc = self.procs[pi]
		print str(pi) + ':\t' + str(proc.loc0)

	def print_edges(self,pi,dim):
		proc = self.procs[pi]
		if dim == 0:
			print str(pi) + ' (x):\t' + str(proc.loc1x)
		else:
			print str(pi) + ' (y):\t' + str(proc.loc1y)

pc = ParaCube(24,2,4)

#print 'results for 24 procs'
#print '0:\t[19 17  1  3  2 23 21 -1'
#pc.print_neighbours(0)
#print '1:\t[17 -1  9  8  3  2  0 19'
#pc.print_neighbours(1)
#print '2:\t[ 0  1  3  5  4 -1 23 21'
#pc.print_neighbours(2)
#print '3:\t[ 1  9  8 -1  5  4  2  0'
#pc.print_neighbours(3)
#print '4:\t[ 2  3  5  7  6 22 23 -1'
#pc.print_neighbours(4)
#print '5:\t[ 3 -1  8 10  7  6  4  2'
#pc.print_neighbours(5)
#print '6:\t[ 4  5  7 12 14 -1 22 23'
#pc.print_neighbours(6)
#print '7:\t[ 5  8 10 -1 12 14  6  4'
#pc.print_neighbours(7)
#print '8:\t[ 3  1  9 11 10  7  5 -1'
#pc.print_neighbours(8)
#print '9:\t[ 1 -1 17 16 11 10  8  3'
#pc.print_neighbours(9)
#print '10:\t[ 8  9 11 13 12 -1  7  5'
#pc.print_neighbours(10)
#print '11:\t[ 9 17 16 -1 13 12 10  8'
#pc.print_neighbours(11)
#print '12:\t[10 11 13 15 14  6  7 -1'
#pc.print_neighbours(12)
#print '13:\t[11 -1 16 18 15 14 12 10'
#pc.print_neighbours(13)
#print '14:\t[12 13 15 20 22 -1  6  7'
#pc.print_neighbours(14)
#print '15:\t[13 16 18 -1 20 22 14 12'
#pc.print_neighbours(15)
#print '16:\t[11  9 17 19 18 15 13 -1'
#pc.print_neighbours(16)
#print '17:\t[ 9 -1  1  0 19 18 16 11'
#pc.print_neighbours(17)
#print '18:\t[16 17 19 21 20 -1 15 13'
#pc.print_neighbours(18)
#print '19:\t[17  1  0 -1 21 20 18 16'
#pc.print_neighbours(19)
#print '20:\t[18 19 21 23 22 14 15 -1'
#pc.print_neighbours(20)
#print '21:\t[19 -1  0  2 23 22 20 18'
#pc.print_neighbours(21)
#print '22:\t[20 21 24  4  6 -1 14 15'
#pc.print_neighbours(22)
#print '23:\t[21  0  2 -1  4  6 22 20'
#pc.print_neighbours(23)

#pc.print_procs()
#print '0:\t[ 0  1  2  3 16  4  5  6  7 20  8  9 10 11 24 12 13 14 15 28 32 33 34 35 48'
#pc.print_nodes(0)
#print '1:\t[ 16  17  18  19 384  20  21  22  23 147  24  25  26  27 146  28  29  30  31 145  48  49  50  51 144'
#pc.print_nodes(1)
#print '2:\t[32 33 34 35 48 36 37 38 39 52 40 41 42 43 56 44 45 46 47 60 64 65 66 67 80'
#pc.print_nodes(2)
#print '3:\t[ 48  49  50  51 144  52  53  54  55 131  56  57  58  59 130  60  61  62  63 129  80  81  82  83 128'
#pc.print_nodes(3)
#print '\n'
#print '4:\t[ 64  65  66  67  80  68  69  70  71  84  72  73  74  75  88  76  77  78  79  92  96  97  98  99 112'
#pc.print_nodes(4)
#print '5:\t[ 80  81  82  83 128  84  85  86  87 132  88  89  90  91 136  92  93  94  95 140 112 113 114 115 160'
#pc.print_nodes(5)
#print '6:\t[ 96  97  98  99 112 100 101 102 103 116 104 105 106 107 120 108 109 110 111 124 385 236 232 228 224'
#pc.print_nodes(6)
#print '7:\t[112 113 114 115 160 116 117 118 119 164 120 121 122 123 168 124 125 126 127 172 224 204 200 196 192'
#pc.print_nodes(7)
#print '\n'
#print '8:\t[128 129 130 131 144 132 133 134 135 148 136 137 138 139 152 140 141 142 143 156 160 161 162 163 176'
#pc.print_nodes(8)
#print '9:\t[144 145 146 147 384 148 149 150 151 275 152 153 154 155 274 156 157 158 159 273 176 177 178 179 272'
#pc.print_nodes(9)
#print '10:\t[160 161 162 163 176 164 165 166 167 180 168 169 170 171 184 172 173 174 175 188 192 193 194 195 208'
#pc.print_nodes(10)
#print '11:\t[176 177 178 179 272 180 181 182 183 259 184 185 186 187 258 188 189 190 191 257 208 209 210 211 256'
#pc.print_nodes(11)
#print '\n'
#print '12:\t[192 193 194 195 208 196 197 198 199 212 200 201 202 203 216 204 205 206 207 220 224 225 226 227 240'
#pc.print_nodes(12)
#print '13:\t[208 209 210 211 256 212 213 214 215 260 216 217 218 219 264 220 221 222 223 268 240 241 242 243 288'
#pc.print_nodes(13)
#print '14:\t[224 225 226 227 240 228 229 230 231 244 232 233 234 235 248 236 237 238 239 252 385 364 360 356 352'
#pc.print_nodes(14)
#print '15:\t[240 241 242 243 288 244 245 246 247 292 248 249 250 251 296 252 253 254 255 300 352 332 328 324 320'
#pc.print_nodes(15)
#print '\n'
#print '16:\t[256 257 258 259 272 260 261 262 263 276 264 265 266 267 280 268 269 270 271 284 288 289 290 291 304'
#pc.print_nodes(16)
#print '17:\t[272 273 274 275 384 276 277 278 279  19 280 281 282 283  18 284 285 286 287  17 304 305 306 307  16'
#pc.print_nodes(17)
#print '18:\t[288 289 290 291 304 292 293 294 295 308 296 297 298 299 312 300 301 302 303 316 320 321 322 323 336'
#pc.print_nodes(18)
#print '19:\t[304 305 306 307  16 308 309 310 311   3 312 313 314 315   2 316 317 318 319   1 336 337 338 339   0'
#pc.print_nodes(19)
#print '\n'
#print '20:\t[320 321 322 323 336 324 325 326 327 340 328 329 330 331 344 332 333 334 335 348 352 353 354 355 368'
#pc.print_nodes(20)
#print '21:\t[336 337 338 339   0 340 341 342 343   4 344 345 346 347   8 348 349 350 351  12 368 369 370 371  32'
#pc.print_nodes(21)
#print '22:\t[352 353 354 355 368 356 357 358 359 372 360 361 362 363 376 364 365 366 367 380 385 108 104 100  96'
#pc.print_nodes(22)
#print '23:\t[368 369 370 371  32 372 373 374 375  36 376 377 378 379  40 380 381 382 383  44  96  76  72  68  64'
#pc.print_nodes(23)

print '0:\t[ 0  1  2  3 16  4  5  6  7 20  8  9 10 11 24 12 13 14 15 28'
pc.print_edges(0,0)
print '0:\t[384 385 386 387 388 389 390 391 396 397 398 399 400 401 402 403 416 417 418 419'
pc.print_edges(0,1)
print '1:\t[ 16  17  18  19 531  20  21  22  23 530  24  25  26  27 529  28  29  30  31 528'
pc.print_edges(1,0)
print '1:\t[400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 432 433 434 435'
pc.print_edges(1,1)
print '2:\t[32 33 34 35 48 36 37 38 39 52 40 41 42 43 56 44 45 46 47 60'
pc.print_edges(2,0)
print '2:\t[416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 448 449 450 451'
pc.print_edges(2,1)
print '3:\t[ 48  49  50  51 515  52  53  54  55 514  56  57  58  59 513  60  61  62  63 512'
pc.print_edges(3,0)
print '3:\t[432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 464 465 466 467'
pc.print_edges(3,1)
print '\n'
print '4:\t[64 65 66 67 80 68 69 70 71 84 72 73 74 75 88 76 77 78 79 92'
pc.print_edges(4,0)
print '4:\t[448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 480 481 482 483'
pc.print_edges(4,1)
print '5:\t[ 80  81  82  83 128  84  85  86  87 132  88  89  90  91 136  92  93  94  95 140'
pc.print_edges(5,0)
print '5:\t[464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 496 497 498 499'
pc.print_edges(5,1)

#pc.print_edges()
