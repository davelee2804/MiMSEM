#!/usr/bin/env python

import sys
import numpy as np

class Proc:
	def __init__(self,pn,nxg,nxl,pix,piy,npx,n_procs,path):
		self.polyDeg = int(pn)
		self.nElsXFace = int(nxg)
		self.nElsXProc = int(nxl)
		self.procX = int(pix)
		self.procY = int(piy)
		self.nProcsX = int(npx)
		self.nProcs = int(n_procs)
		self.procID = piy*npx + pix
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
		self.n0l = (self.nDofsXProc)*(self.nDofsXProc)
		self.n1xl = (self.nDofsXProc)*(self.nDofsXProc)
		self.n1yl = (self.nDofsXProc)*(self.nDofsXProc)
		self.n2l = (self.nDofsXProc)*(self.nDofsXProc)

		# number of global nodes/edges/faces (on this processor)
		self.n0g = (self.nDofsXProc+1)*(self.nDofsXProc+1)
		self.n1xg = (self.nDofsXProc+1)*(self.nDofsXProc)
		self.n1yg = (self.nDofsXProc)*(self.nDofsXProc+1)
		self.n2g = (self.nDofsXProc)*(self.nDofsXProc)

		# global offsets
		self.shift0 = self.procY*self.nDofsXProc*self.nDofsXFace + self.procX*self.nDofsXProc
		self.shift1 = 2*(self.procY*self.nProcsX+self.procX)*self.nDofsXProc*self.nDofsXProc
		self.shift2 = (self.procY*self.nProcsX+self.procX)*self.nDofsXProc*self.nDofsXProc

		# global index arrays
		self.loc0 = np.zeros(self.n0g,dtype=np.int32)
		self.loc1x = np.zeros(self.n1xg,dtype=np.int32)
		self.loc1y = np.zeros(self.n1yg,dtype=np.int32)
		self.loc2 = np.zeros(self.n2g,dtype=np.int32)

		# global indices of local nodes
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc):
				self.loc0[iy*(self.nDofsXProc+1) + ix] = self.shift0 + iy*self.nDofsXFace + ix

		# global indices of local x normal edges
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc):
				ey = iy // self.polyDeg
				py = iy % self.polyDeg
				ex = ix // self.polyDeg
				px = ix % self.polyDeg
				self.loc1x[iy*(self.nDofsXProc+1) + ix] = self.shift1 + 2*((ey*self.nElsXProc + ex)*self.polyDeg*self.polyDeg + py*self.polyDeg+px)

		# global indices of local y normal edges
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc):
				ey = iy // self.polyDeg
				py = iy % self.polyDeg
				ex = ix // self.polyDeg
				px = ix % self.polyDeg
				self.loc1y[iy*(self.nDofsXProc) + ix] = self.shift1 + 2*((ey*self.nElsXProc + ex)*self.polyDeg*self.polyDeg + py*self.polyDeg+px) + 1

		# global indices of local faces
		for iy in np.arange(self.nDofsXProc):
			for ix in np.arange(self.nDofsXProc):
				ey = iy // self.polyDeg
				py = iy % self.polyDeg
				ex = ix // self.polyDeg
				px = ix % self.polyDeg
				self.loc2[iy*self.nDofsXProc + ix] = self.shift2 + (ey*self.nElsXProc + ex)*self.polyDeg*self.polyDeg + py*self.polyDeg+px

	# define global indices for remote nodes/edges/faces
	def buildGlobalArrays(self):
		east = self.EE
		north = self.NN
		if self.EN != None:
			northEast = self.EN
		else:
			northEast = None
		if self.SE != None:
			southEast = self.SE
		else:
			southEast = None
		if self.NW != None:
			northWest = self.NW
		else:
			northWest = None

		# 1. east and north processor neighbours are on the same face
		gInds0 = east.getW0(+1)
		for iy in np.arange(self.nDofsXProc+1):
			self.loc0[iy*(self.nDofsXProc+1) + (self.nDofsXProc)] = gInds0[iy]

		gInds0 = north.getS0(+1)
		for ix in np.arange(self.nDofsXProc+1):
			self.loc0[(self.nDofsXProc)*(self.nDofsXProc+1) + ix] = gInds0[ix]

		gInds0 = northEast.getS0(+1)
		self.loc0[(self.nDofsXProc+1)*(self.nDofsXProc+1) - 1] = gInds0[0]

		# now do the edges (north and south procs on same face)
		gInds1x = east.getW1(+1)
		for iy in np.arange(self.nDofsXProc):
			self.loc1x[iy*(self.nDofsXProc+1) + (self.nDofsXProc)] = gInds1x[iy]

		gInds1y = north.getS1(+1)
		for ix in np.arange(self.nDofsXProc):
			self.loc1y[(self.nDofsXProc)*(self.nDofsXProc) + ix] = gInds1y[ix]

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

class ParaBox:
	def __init__(self,n_procs,pn,nx,path):
		self.np = n_procs	# total number of processors
		self.pn = pn		# polynomial degree
		self.nx = nx		# number of elements across a face (global)
		self.path = path

		npx = int(np.sqrt(n_procs))
		self.npx = npx
		print('no. procs in each dimsion: ' + str(self.npx))

		self.procs = [None]*(npx*npx)

		for pi in np.arange(npx*npx):
			self.procs[pi] = Proc(self.pn,self.nx,self.nx//self.npx,pi%npx,pi//npx,npx,self.np,self.path)

		# periodic connectivity of processors
		for pi in np.arange(npx*npx):
			px = pi%npx
			py = pi//npx

			pa = (px+0)%npx
			pb = (py-1)%npx
			self.procs[pi].SS = self.procs[pb*npx+pa]

			pa = (px+1)%npx
			pb = (py-1)%npx
			self.procs[pi].SE = self.procs[pb*npx+pa]

			pa = (px+1)%npx
			pb = (py+0)%npx
			self.procs[pi].EE = self.procs[pb*npx+pa]

			pa = (px+1)%npx
			pb = (py+1)%npx
			self.procs[pi].EN = self.procs[pb*npx+pa]

			pa = (px+0)%npx
			pb = (py+1)%npx
			self.procs[pi].NN = self.procs[pb*npx+pa]

			pa = (px-1)%npx
			pb = (py+1)%npx
			self.procs[pi].NW = self.procs[pb*npx+pa]

			pa = (px-1)%npx
			pb = (py+0)%npx
			self.procs[pi].WW = self.procs[pb*npx+pa]

			pa = (px-1)%npx
			pb = (py-1)%npx
			self.procs[pi].WS = self.procs[pb*npx+pa]

		# build the global index maps for the nodes/edges/faces on a given processor
		for pi in np.arange(self.np):
			self.procs[pi].buildLocalArrays()
		for pi in np.arange(self.np):
			self.procs[pi].buildGlobalArrays()

	def print_nodes(self,pi):
		proc = self.procs[pi]
		np.savetxt(self.path+'/input/nodes_%.4u'%pi + '.txt', proc.loc0, fmt='%u')

	def print_edges(self,pi,dim):
		proc = self.procs[pi]
		if dim == 0:
			np.savetxt(self.path+'/input/edges_x_%.4u'%pi + '.txt', proc.loc1x, fmt='%u')
		else:
			np.savetxt(self.path+'/input/edges_y_%.4u'%pi + '.txt', proc.loc1y, fmt='%u')

	def print_faces(self,pi):
		proc = self.procs[pi]
		np.savetxt(self.path+'/input/faces_%.4u'%pi + '.txt', proc.loc2, fmt='%u')
