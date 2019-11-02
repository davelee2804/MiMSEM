#!/usr/bin/env python

import sys
import numpy as np

# container class for the processor on a given side of the
# target processor and its relative coordinate system transformation
class Side:
	def __init__(self,proc,axis):
		self.proc = proc
		self.axis = axis

class Proc:
	def __init__(self,pn,nxg,nxl,pix,piy,npx,fi,n_procs,path):
		self.polyDeg = pn
		self.nElsXFace = nxg
		self.nElsXProc = nxl
		self.procX = pix
		self.procY = piy
		self.nProcsX = npx
		self.faceID = fi
		self.nProcs = n_procs
		self.procID = fi*npx*npx + piy*npx + pix
		self.path = path

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
		#self.shift1 = 2*(self.faceID*self.nDofsXFace*self.nDofsXFace + self.procY*self.nDofsXProc*self.nDofsXFace + self.procX*self.nDofsXProc)
		#self.shift2 = self.faceID*self.nDofsXFace*self.nDofsXFace + self.procY*self.nDofsXProc*self.nDofsXFace + self.procX*self.nDofsXProc
		self.shift1 = 2*(self.faceID*self.nDofsXFace*self.nDofsXFace + (self.procY*self.nProcsX+self.procX)*self.nDofsXProc*self.nDofsXProc)
		self.shift2 = self.faceID*self.nDofsXFace*self.nDofsXFace + (self.procY*self.nProcsX+self.procX)*self.nDofsXProc*self.nDofsXProc

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
				#self.loc1x[iy*(self.nDofsXProc+1) + ix] = self.shift1 + 2*(iy*self.nDofsXFace + ix)
				ey = iy / self.polyDeg
				py = iy % self.polyDeg
				ex = ix / self.polyDeg
				px = ix % self.polyDeg
				self.loc1x[iy*(self.nDofsXProc+1) + ix] = self.shift1 + 2*((ey*self.nElsXProc + ex)*self.polyDeg*self.polyDeg + py*self.polyDeg+px)

		# global indices of local y normal edges
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc):
				#self.loc1y[iy*(self.nDofsXProc) + ix] = self.shift1 + 2*(iy*self.nDofsXFace + ix) + 1
				ey = iy / self.polyDeg
				py = iy % self.polyDeg
				ex = ix / self.polyDeg
				px = ix % self.polyDeg
				self.loc1y[iy*(self.nDofsXProc) + ix] = self.shift1 + 2*((ey*self.nElsXProc + ex)*self.polyDeg*self.polyDeg + py*self.polyDeg+px) + 1

		# global indices of local faces
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc):
				#self.loc2[iy*(self.nDofsXProc) + ix] = self.shift2 + iy*self.nDofsXFace + ix
				ey = iy / self.polyDeg
				py = iy % self.polyDeg
				ex = ix / self.polyDeg
				px = ix % self.polyDeg
				self.loc2[iy*self.nDofsXProc + ix] = self.shift2 + (ey*self.nElsXProc + ex)*self.polyDeg*self.polyDeg + py*self.polyDeg+px

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
		np.savetxt(self.path+'/input/local_sizes_%.4u'%self.procID + '.txt', a, fmt='%u')

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
	def __init__(self,n_procs,pn,nx,path):
		self.np = n_procs	# total number of processors
		self.pn = pn		# polynomial degree
		self.nx = nx		# number of elements across a face (global)
		self.path = path

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
				self.procs[pi] = Proc(self.pn,self.nx,self.nx/self.npx,pj%npx,pj/npx,npx,fi,self.np,self.path)
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

		#print str(pi) + ':\t' + str(neighbours)

	def print_nodes(self,pi):
		proc = self.procs[pi]
		#print str(pi) + ':\t' + str(proc.loc0)
		np.savetxt(self.path+'/input/nodes_%.4u'%pi + '.txt', proc.loc0, fmt='%u')

	def print_edges(self,pi,dim):
		proc = self.procs[pi]
		if dim == 0:
			#print str(pi) + ' (x):\t' + str(proc.loc1x)
			np.savetxt(self.path+'/input/edges_x_%.4u'%pi + '.txt', proc.loc1x, fmt='%u')
		else:
			#print str(pi) + ' (y):\t' + str(proc.loc1y)
			np.savetxt(self.path+'/input/edges_y_%.4u'%pi + '.txt', proc.loc1y, fmt='%u')

	def print_faces(self,pi):
		proc = self.procs[pi]
		#print str(pi) + ':\t' + str(proc.loc2)
		np.savetxt(self.path+'/input/faces_%.4u'%pi + '.txt', proc.loc2, fmt='%u')
