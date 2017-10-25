#!/usr/bin/env python

import numpy as np

# container class for the processor on a given side of the
# target processor and its relative coordinate system transformation
class Side:
	def __init__(self,proc,axis):
		self.proc = proc
		self.axis = axis

class Proc:
	def __init__(self,pn,nxg,nxl,pix,piy,npx,fi,n_procs):
		self.polyDeg = pn
		self.nElsXFace = nxg
		self.nElsXProc = nxl
		self.procX = pix
		self.procY = piy
		self.nProcsX = npx
		self.faceID = fi
		self.nProcs = n_procs
		self.procID = fi*npx*npx + piy*npx + pix

		self.nDofsXProc = self.polyDeg*self.nElsXProc
		self.nDofsXFace = self.polyDeg*self.nElsXFace

		self.xShiftLoc = (self.nDofsXProc+1)*(self.nDofsXProc)

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
		self.inds0 = np.zeros((self.polyDeg+1)*(self.polyDeg+1),dtype=np.int32)
		self.inds1x = np.zeros((self.polyDeg+1)*(self.polyDeg),dtype=np.int32)
		self.inds1y = np.zeros((self.polyDeg)*(self.polyDeg+1),dtype=np.int32)
		self.inds2 = np.zeros((self.polyDeg)*(self.polyDeg),dtype=np.int32)

		# global index arrays for sides of processor region
		self.side0 = np.zeros(self.nDofsXProc+1,dtype=np.int32)
		self.side1 = np.zeros(self.nDofsXProc,dtype=np.int32)

	# define global indices for local nodes/edges/faces
	def buildLocalArrays(self):
		# number of local nodes/edges/faces
		self.n0lavg = (self.nDofsXProc)*(self.nDofsXProc)
		self.n0l = self.n0lavg
		# account for the two hanging nodes on the cube
		# note: these two hanging nodes have the last two global indices
		if self.faceID == 0 and self.SE == None:
			self.n0l = self.n0lavg + 1
		if self.faceID == 1 and self.NW == None:
			self.n0l = self.n0lavg + 1
		self.n1xl = (self.nDofsXProc)*(self.nDofsXProc)
		self.n1yl = (self.nDofsXProc)*(self.nDofsXProc)
		self.n2l = (self.nDofsXProc)*(self.nDofsXProc)

		# number of global nodes/edges/faces (on this processor)
		self.n0g = (self.nDofsXProc+1)*(self.nDofsXProc+1)
		self.n1xg = (self.nDofsXProc+1)*(self.nDofsXProc)
		self.n1yg = (self.nDofsXProc)*(self.nDofsXProc+1)
		self.n2g = (self.nDofsXProc)*(self.nDofsXProc)

		# global offsets
		self.shift0 = self.faceID*self.nDofsXFace*self.nDofsXFace + self.procY*self.nDofsXProc*self.nDofsXFace + self.procX*self.nDofsXProc
		self.shift1x = self.faceID*self.nDofsXFace*self.nDofsXFace + self.procY*self.nDofsXProc*self.nDofsXFace + self.procX*self.nDofsXProc
		self.shift1y = 6*self.nDofsXFace*self.nDofsXFace + self.shift1x
		self.shift2 = self.faceID*self.nDofsXFace*self.nDofsXFace + self.procY*self.nDofsXProc*self.nDofsXFace + self.procX*self.nDofsXProc

		# global index arrays
		self.loc0 = np.zeros(self.n0g,dtype=np.int32)
		self.loc1x = np.zeros(self.n1xg,dtype=np.int32)
		self.loc1y = np.zeros(self.n1yg,dtype=np.int32)
		self.loc2 = np.zeros(self.n2g,dtype=np.int32)

		# incidence relations
		self.edgeToFace = np.zeros((4,self.n2g),dtype=np.int32)
		self.nodeToEdgeX = np.zeros((2,self.n1xg),dtype=np.int32)
		self.nodeToEdgeY = np.zeros((2,self.n1yg),dtype=np.int32)

		# global indices of local nodes
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc):
				self.loc0[iy*(self.nDofsXProc+1) + ix] = self.shift0 + iy*self.nDofsXFace + ix

		# hanging nodes
		if self.faceID == 0 and self.SE == None:
			self.loc0[self.nDofsXProc] = self.nProcs*self.n0lavg
		if self.faceID == 1 and self.NW == None:
			self.loc0[(self.nDofsXProc)*(self.nDofsXProc+1)] = self.nProcs*self.n0lavg + 1

		# global indices of local x normal edges
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc):
				self.loc1x[iy*(self.nDofsXProc+1) + ix] = self.shift1x + iy*self.nDofsXFace + ix

		# global indices of local y normal edges
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc):
				self.loc1y[iy*(self.nDofsXProc) + ix] = self.shift1y + iy*self.nDofsXFace + ix

		# global indices of local faces
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc):
				self.loc2[iy*(self.nDofsXProc) + ix] = self.shift2 + iy*self.nDofsXFace + ix

	# define global indices for remote nodes/edges/faces
	def buildGlobalArrays(self):
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
			for iy in np.arange(self.nDofsXProc+1):
				self.loc0[iy*(self.nDofsXProc+1) + (self.nDofsXProc)] = gInds0[iy]

			gInds0 = north.getS0(+1)
			for ix in np.arange(self.nDofsXProc+1):
				self.loc0[(self.nDofsXProc)*(self.nDofsXProc+1) + ix] = gInds0[ix]

			gInds0 = northEast.getS0(+1)
			self.loc0[(self.nDofsXProc+1)*(self.nDofsXProc+1) - 1] = gInds0[0]

		# 2. north proc on same axis and east proc rotated -pi/2
		elif nAxis[1][1] == +1 and eAxis[0][1] == +1:
			gInds0 = north.getS0(+1)
			for ix in np.arange(self.nDofsXProc+1):
				self.loc0[(self.nDofsXProc)*(self.nDofsXProc+1) + ix] = gInds0[ix]

			gInds0 = east.getS0(-1)
			for iy in np.arange(self.nDofsXProc+1):
				self.loc0[iy*(self.nDofsXProc+1) + (self.nDofsXProc)] = gInds0[iy]

			if southEast == None:
				self.loc0[(self.nDofsXProc)] = self.nProcs*self.n0lavg
			else:
				gInds0 = southEast.getS0(+1)
				self.loc0[(self.nDofsXProc)] = gInds0[0]
				
		# 3. east proc on same axis and noth proc rotated +pi/2
		elif eAxis[0][0] == +1 and nAxis[1][0] == +1:
			gInds0 = east.getW0(+1)
			for iy in np.arange(self.nDofsXProc+1):
				self.loc0[iy*(self.nDofsXProc+1) + (self.nDofsXProc)] = gInds0[iy]

			gInds0 = north.getW0(-1)
			for ix in np.arange(self.nDofsXProc+1):
				self.loc0[(self.nDofsXProc)*(self.nDofsXProc+1) + ix] = gInds0[ix]

			if northWest == None:
				self.loc0[(self.nDofsXProc)*(self.nDofsXProc+1)] = self.nProcs*self.n0lavg + 1
			else:
				gInds0 = northWest.getS0(+1)
				self.loc0[(self.nDofsXProc)*(self.nDofsXProc+1)] = gInds0[0]

		else:
			print 'adjacency error (2.0)'

		# now do the edges (north and south procs on same face)
		if eAxis[0][0] == +1 and nAxis[1][1] == +1:
			gInds1x = east.getW1(+1)
			for iy in np.arange(self.nDofsXProc):
				self.loc1x[iy*(self.nDofsXProc+1) + (self.nDofsXProc)] = gInds1x[iy]

			gInds1y = north.getS1(+1)
			for ix in np.arange(self.nDofsXProc):
				self.loc1y[(self.nDofsXProc)*(self.nDofsXProc) + ix] = gInds1y[ix]

		# 2. north proc on same axis and east proc rotated -pi/2
		elif nAxis[1][1] == +1 and eAxis[0][1] == +1:
			gInds1y = north.getS1(+1)
			for ix in np.arange(self.nDofsXProc):
				self.loc1y[(self.nDofsXProc)*(self.nDofsXProc) + ix] = gInds1y[ix]

			gInds1x = east.getS1(-1)
			for iy in np.arange(self.nDofsXProc):
				self.loc1x[iy*(self.nDofsXProc+1) + self.nDofsXProc] = gInds1x[iy]

		# 3. east proc on same axis and noth proc rotated +pi/2
		elif eAxis[0][0] == +1 and nAxis[1][0] == +1:
			gInds1x = east.getW1(+1)
			for iy in np.arange(self.nDofsXProc):
				self.loc1x[iy*(self.nDofsXProc+1) + self.nDofsXProc] = gInds1x[iy]

			gInds1y = north.getW1(-1)
			for ix in np.arange(self.nDofsXProc):
				self.loc1y[(self.nDofsXProc)*(self.nDofsXProc) + ix] = gInds1y[ix]

		else:
			print 'adjacency error (2.1)'

	# node to edge and edge to face incidence relations
	def buildIncidence(self):
		# edge to face incidence
		kk = 0
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc):
				self.edgeToFace[0,kk] = self.loc1x[iy*(self.nDofsXProc+1)+ix]
				self.edgeToFace[1,kk] = self.loc1x[iy*(self.nDofsXProc+1)+ix+1]
				self.edgeToFace[2,kk] = self.loc1y[iy*(self.nDofsXProc)+ix]
				self.edgeToFace[3,kk] = self.loc1y[(iy+1)*(self.nDofsXProc)+ix]
				kk = kk + 1

		# node to edge incidence (x-normal edges)
		kk = 0
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc+1):
				self.nodeToEdgeX[0,kk] = self.loc0[iy*(self.nDofsXProc+1)+ix]
				self.nodeToEdgeX[1,kk] = self.loc0[(iy+1)*(self.nDofsXProc+1)+ix]

		# node to edge incidence (y-normal edges)
		kk = 0
		for iy in np.arange(self.nDofsXProc+1):
			for ix in np.arange(self.nDofsXProc):
				self.nodeToEdgeY[0,kk] = self.loc0[iy*(self.nDofsXProc+1)+ix]
				self.nodeToEdgeY[1,kk] = self.loc0[iy*(self.nDofsXProc+1)+ix+1]

	# return the local indices for the nodes of given element
	def elementToLocal0(self,ex,ey):
		kk = 0
		for iy in np.arange(self.polyDeg+1):
			for ix in np.arange(self.polyDeg+1):
				self.inds0[kk] = (ey*self.polyDeg+iy)*(self.nDofsXProc+1) + ex*self.polyDeg + ix
				kk = kk + 1

		return self.inds0

	# return the local indices for the x normal edges of given element
	def elementToLocal1x(self,ex,ey):
		kk = 0
		for iy in np.arange(self.polyDeg):
			for ix in np.arange(self.polyDeg+1):
				self.inds1x[kk] = (ey*self.polyDeg+iy)*(self.nDofsXProc+1) + ex*self.polyDeg + ix
				kk = kk + 1

		return self.inds1x

	# return the local indices for the y normal edges of given element
	def elementToLocal1y(self,ex,ey):
		kk = 0
		for iy in np.arange(self.polyDeg+1):
			for ix in np.arange(self.polyDeg):
				self.inds1y[kk] = (ey*self.polyDeg+iy)*(self.nDofsXProc) + ex*self.polyDeg + ix
				kk = kk + 1

		return self.inds1y + self.xShiftLoc

	# return the local indices for the faces of given element
	def elementToLocal2(self,ex,ey):
		kk = 0
		for iy in np.arange(self.polyDeg):
			for ix in np.arange(self.polyDeg):
				self.inds2[kk] = (ey*self.polyDeg+iy)*(self.nDofsXProc) + ex*self.polyDeg + ix
				kk = kk + 1

		return self.inds2

	# return the global node indices on the west side of the processor region
	def getW0(self,orient):
		for iy in np.arange(self.nDofsXProc+1):
			self.side0[iy] = self.loc0[iy*(self.nDofsXProc+1)]

		if orient == -1:
			self.side0[:] = self.side0[::-1]

		return self.side0

	# return the global edge indices on the west side of the processor region
	def getW1(self,orient):
		for iy in np.arange(self.nDofsXProc):
			self.side1[iy] = self.loc1x[iy*(self.nDofsXProc+1)]

		if orient == -1:
			self.side1 = self.side1[::-1]

		return self.side1

	# return the global node indices on the south side of the processor region
	def getS0(self,orient):
		for ix in np.arange(self.nDofsXProc+1):
			self.side0[ix] = self.loc0[ix]

		if orient == -1:
			self.side0[:] = self.side0[::-1]

		return self.side0

	# return the global edge indices on the south side of the processor region
	def getS1(self,orient):
		for ix in np.arange(self.nDofsXProc):
			self.side1[ix] = self.loc1y[ix]

		if orient == -1:
			self.side1 = self.side1[::-1]

		return self.side1

	def writeLocalSizes(self):
		a = np.zeros(4,dtype=np.int32)
		a[0] = self.n0l
		a[1] = self.n1xl
		a[2] = self.n1yl
		a[3] = self.n2l
		np.savetxt('local_sizes_%.4u'%self.procID + '.txt', a, fmt='%u')

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
				self.procs[pi] = Proc(self.pn,self.nx,self.nx/self.npx,pj%npx,pj/npx,npx,fi,self.np)
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

		neighbours[0] = proc.SS.proc.procID
		if proc.SE == None:
			neighbours[1] = -1
		else:
			neighbours[1] = proc.SE.proc.procID
		neighbours[2] = proc.EE.proc.procID
		if proc.EN == None:
			neighbours[3] = -1
		else:
			neighbours[3] = proc.EN.proc.procID
		neighbours[4] = proc.NN.proc.procID
		if proc.NW == None:
			neighbours[5] = -1
		else:
			neighbours[5] = proc.NW.proc.procID
		neighbours[6] = proc.WW.proc.procID
		if proc.WS == None:
			neighbours[7] = -1
		else:
			neighbours[7] = proc.WS.proc.procID

		print str(pi) + ':\t' + str(neighbours)

	def print_nodes(self,pi):
		proc = self.procs[pi]
		print str(pi) + ':\t' + str(proc.loc0)
		np.savetxt('nodes_%.4u'%pi + '.txt', proc.loc0, fmt='%u')

	def print_edges(self,pi,dim):
		proc = self.procs[pi]
		if dim == 0:
			print str(pi) + ' (x):\t' + str(proc.loc1x)
			np.savetxt('edges_x_%.4u'%pi + '.txt', proc.loc1x, fmt='%u')
		else:
			print str(pi) + ' (y):\t' + str(proc.loc1y)
			np.savetxt('edges_y_%.4u'%pi + '.txt', proc.loc1y, fmt='%u')

	def print_faces(self,pi):
		proc = self.procs[pi]
		print str(pi) + ':\t' + str(proc.loc2)
		np.savetxt('faces_%.4u'%pi + '.txt', proc.loc2, fmt='%u')

n_procs = 6
pc = ParaCube(n_procs,3,8)

for pi in np.arange(n_procs):
	pc.print_nodes(pi)
	pc.print_edges(pi,0)
	pc.print_edges(pi,1)
	pc.print_faces(pi)
	pc.procs[pi].writeLocalSizes()

'''
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

print '\nnode topology\n'
#pc.print_procs()
print '0:\t[ 0  1  2  3  4  8  9 10 11 12 16 17 18 19 20 24 25 26 27 28 32 33 34 35 36'
pc.print_nodes(0)
print '1:\t[  4   5   6   7 384  12  13  14  15 135  20  21  22  23 134  28  29  30  31 133  36  37  38  39 132'
pc.print_nodes(1)
print '2:\t[32 33 34 35 36 40 41 42 43 44 48 49 50 51 52 56 57 58 59 60 64 65 66 67 68'
pc.print_nodes(2)
print '3:\t[ 36  37  38  39 132  44  45  46  47 131  52  53  54  55 130  60  61  62  63 129  68  69  70  71 128'
pc.print_nodes(3)
print '\n'
print '4:\t[ 64  65  66  67  68  72  73  74  75  76  80  81  82  83  84  88  89  90  91  92  96  97  98  99 100'
pc.print_nodes(4)
print '5:\t[ 68  69  70  71 128  76  77  78  79 136  84  85  86  87 144  92  93  94  95 152 100 101 102 103 160'
pc.print_nodes(5)
print '6:\t[ 96  97  98  99 100 104 105 106 107 108 112 113 114 115 116 120 121 122 123 124 385 248 240 232 224'
pc.print_nodes(6)
print '7:\t[100 101 102 103 160 108 109 110 111 168 116 117 118 119 176 124 125 126 127 184 224 216 208 200 192'
pc.print_nodes(7)
print '\n'
print '8:\t[128 129 130 131 132 136 137 138 139 140 144 145 146 147 148 152 153 154 155 156 160 161 162 163 164'
pc.print_nodes(8)
print '9:\t[132 133 134 135 384 140 141 142 143 263 148 149 150 151 262 156 157 158 159 261 164 165 166 167 260'
pc.print_nodes(9)
print '10:\t[160 161 162 163 164 168 169 170 171 172 176 177 178 179 180 184 185 186 187 188 192 193 194 195 196'
pc.print_nodes(10)
print '11:\t[164 165 166 167 260 172 173 174 175 259 180 181 182 183 258 188 189 190 191 257 196 197 198 199 256'
pc.print_nodes(11)
print '\n'
print '12:\t[192 193 194 195 196 200 201 202 203 204 208 209 210 211 212 216 217 218 219 220 224 225 226 227 228'
pc.print_nodes(12)
print '13:\t[196 197 198 199 256 204 205 206 207 264 212 213 214 215 272 220 221 222 223 280 228 229 230 231 288'
pc.print_nodes(13)
print '14:\t[224 225 226 227 228 232 233 234 235 236 240 241 242 243 244 248 249 250 251 252 385 376 368 360 352'
pc.print_nodes(14)
print '15:\t[228 229 230 231 288 236 237 238 239 296 244 245 246 247 304 252 253 254 255 312 352 344 336 328 320'
pc.print_nodes(15)
print '\n'
print '16:\t[256 257 258 259 260 264 265 266 267 268 272 273 274 275 276 280 281 282 283 284 288 289 290 291 292'
pc.print_nodes(16)
print '17:\t[260 261 262 263 384 268 269 270 271   7 276 277 278 279   6 284 285 286 287   5 292 293 294 295   4'
pc.print_nodes(17)
print '18:\t[288 289 290 291 292 296 297 298 299 300 304 305 306 307 308 312 313 314 315 316 320 321 322 323 324'
pc.print_nodes(18)
print '19:\t[292 293 294 295   4 300 301 302 303   3 308 309 310 311   2 316 317 318 319   1 324 325 326 327   0'
pc.print_nodes(19)
print '\n'
print '20:\t[320 321 322 323 324 328 329 330 331 332 336 337 338 339 340 344 345 346 347 348 352 353 354 355 356'
pc.print_nodes(20)
print '21:\t[324 325 326 327   0 332 333 334 335   8 340 341 342 343  16 348 349 350 351  24 356 357 358 359  32'
pc.print_nodes(21)
print '22:\t[352 353 354 355 356 360 361 362 363 364 368 369 370 371 372 376 377 378 379 380 385 120 112 104  96'
pc.print_nodes(22)
print '23:\t[356 357 358 359  32 364 365 366 367  40 372 373 374 375  48 380 381 382 383  56  96  88  80  72  64'
pc.print_nodes(23)

print '\nedge topology\n'
#pc.print_edges()
print '0:\t[ 0  1  2  3  4  8  9 10 11 12 16 17 18 19 20 24 25 26 27 28'
pc.print_edges(0,0)
print '0:\t[384 385 386 387 392 393 394 395 400 401 402 403 408 409 410 411 416 417 418 419'
pc.print_edges(0,1)
print '1:\t[  4   5   6   7 519  12  13  14  15 518  20  21  22  23 517  28  29  30  31 516'
pc.print_edges(1,0)
print '1:\t[388 389 390 391 396 397 398 399 404 405 406 407 412 413 414 415 420 421 422 423'
pc.print_edges(1,1)
print '2:\t[32 33 34 35 36 40 41 42 43 44 48 49 50 51 52 56 57 58 59 60'
pc.print_edges(2,0)
print '2:\t[416 417 418 419 424 425 426 427 432 433 434 435 440 441 442 443 448 449 450 451'
pc.print_edges(2,1)
print '3:\t[ 36  37  38  39 515  44  45  46  47 514  52  53  54  55 513  60  61  62  63 512'
pc.print_edges(3,0)
print '3:\t[420 421 422 423 428 429 430 431 436 437 438 439 444 445 446 447 452 453 454 455'
pc.print_edges(3,1)
print '\n'
print '4:\t[64 65 66 67 68 72 73 74 75 76 80 81 82 83 84 88 89 90 91 92'
pc.print_edges(4,0)
print '4:\t[448 449 450 451 456 457 458 459 464 465 466 467 472 473 474 475 480 481 482 483'
pc.print_edges(4,1)
print '5:\t[ 68  69  70  71 128  76  77  78  79 136  84  85  86  87 144  92  93  94  95 152'
pc.print_edges(5,0)
print '5:\t[452 453 454 455 460 461 462 463 468 469 470 471 476 477 478 479 484 485 486 487'
pc.print_edges(5,1)
print '6:\t[ 96  97  98  99 100 104 105 106 107 108 112 113 114 115 116 120 121 122 123 124'
pc.print_edges(6,0)
print '6:\t[480 481 482 483 488 489 490 491 496 497 498 499 504 505 506 507 248 240 232 224'
pc.print_edges(6,1)
print '7:\t[100 101 102 103 160 108 109 110 111 168 116 117 118 119 176 124 125 126 127 184'
pc.print_edges(7,0)
print '7:\t[484 485 486 487 492 493 494 495 500 501 502 503 508 509 510 511 216 208 200 192'
pc.print_edges(7,1)
print '\n'
print '8:\t[128 129 130 131 132 136 137 138 139 140 144 145 146 147 148 152 153 154 155 156'
pc.print_edges(8,0)
print '8:\t[512 513 514 515 520 521 522 523 528 529 530 531 536 537 538 539 544 545 546 547'
pc.print_edges(8,1)
print '9:\t[132 133 134 135 647 140 141 142 143 646 148 149 150 151 645 156 157 158 159 644'
pc.print_edges(9,0)
print '9:\t[516 517 518 519 524 525 526 527 532 533 534 535 540 541 542 543 548 549 550 551'
pc.print_edges(9,1)
print '10:\t[160 161 162 163 164 168 169 170 171 172 176 177 178 179 180 184 185 186 187 188'
pc.print_edges(10,0)
print '10:\t[544 545 546 547 552 553 554 555 560 561 562 563 568 569 570 571 576 577 578 579'
pc.print_edges(10,1)
print '11:\t[164 165 166 167 643 172 173 174 175 642 180 181 182 183 641 188 189 190 191 640'
pc.print_edges(11,0)
print '11:\t[548 549 550 551 556 557 558 559 564 565 566 567 572 573 574 575 580 581 582 583'
pc.print_edges(11,1)
print '\n'
print '12:\t[192 193 194 195 196 200 201 202 203 204 208 209 210 211 212 216 217 218 219 220'
pc.print_edges(12,0)
print '12:\t[576 577 578 579 584 585 586 587 592 593 594 595 600 601 602 603 608 609 610 611'
pc.print_edges(12,1)
print '13:\t[196 197 198 199 256 204 205 206 207 264 212 213 214 215 272 220 221 222 223 280'
pc.print_edges(13,0)
print '13:\t[580 581 582 583 588 589 590 591 596 597 598 599 604 605 606 607 612 613 614 615'
pc.print_edges(13,1)
print '14:\t[224 225 226 227 228 232 233 234 235 236 240 241 242 243 244 248 249 250 251 252'
pc.print_edges(14,0)
print '14:\t[608 609 610 611 616 617 618 619 624 625 626 627 632 633 634 635 376 368 360 352'
pc.print_edges(14,1)
print '15:\t[228 229 230 231 288 236 237 238 239 296 244 245 246 247 304 252 253 254 255 312'
pc.print_edges(15,0)
print '15:\t[612 613 614 615 620 621 622 623 628 629 630 631 636 637 638 639 344 336 328 320'
pc.print_edges(15,1)
print '\n'
print '16:\t[256 257 258 259 260 264 265 266 267 268 272 273 274 275 276 280 281 282 283 284'
pc.print_edges(16,0)
print '16:\t[640 641 642 643 648 649 650 651 656 657 658 659 664 665 666 667 672 673 674 675'
pc.print_edges(16,1)
print '17:\t[260 261 262 263 391 268 269 270 271 390 276 277 278 279 389 284 285 286 287 388'
pc.print_edges(17,0)
print '17:\t[644 645 646 647 652 653 654 655 660 661 662 663 668 669 670 671 676 677 678 679'
pc.print_edges(17,1)
print '18:\t[288 289 290 291 292 296 297 298 299 300 304 305 306 307 308 312 313 314 315 316'
pc.print_edges(18,0)
print '18:\t[672 673 674 675 680 681 682 683 688 689 690 691 696 697 698 699 704 705 706 707'
pc.print_edges(18,1)
print '19:\t[292 293 294 295 387 300 301 302 303 386 308 309 310 311 385 316 317 318 319 384'
pc.print_edges(19,0)
print '19:\t[676 677 678 679 684 685 686 687 692 693 694 695 700 701 702 703 708 709 710 711'
pc.print_edges(19,1)
print '\n'
'''

