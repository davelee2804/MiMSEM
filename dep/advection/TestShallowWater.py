#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.sparse.linalg as la

from Basis import *
from Topo import *
from BoundaryMat import *
from Mats2D import *
from Assembly import *
from Proj import *
from ShallowWaterEqn import *
from Utils import *

nx = 20
ny = 20
n = 3 # basis order
m = 3 # quadrature order
np1 = n+1
mp1 = m+1
nxn = nx*n
nyn = ny*n
nxm = nx*m
nym = ny*m
shift1Form = nxn*nyn
shiftQuad = nxm*nym

topo = Topo(nx,ny,n)
quad = GaussLobatto(m)
topo_q = Topo(nx,ny,m)

lx = 10.0#2.0*np.pi
ly = 10.0#2.0*np.pi
dx = lx/nx
dy = ly/ny

node = LagrangeNode(n)
Njxi = np.zeros((mp1,np1),dtype=np.float64)
for j in np.arange(np1):
	for i in np.arange(mp1):
		Njxi[i,j] = node.eval(quad.x[i],j)

edge = LagrangeEdge(n)
Ejxi = np.zeros((mp1,n),dtype=np.float64)
for j in np.arange(n):
	for i in np.arange(mp1):
		Ejxi[i,j] = edge.eval(quad.x[i],j)

u2dn = np.zeros((nym,nxm),dtype=np.float64)
v2dn = np.zeros((nym,nxm),dtype=np.float64)
q2dn = np.zeros((nym,nxm),dtype=np.float64)
q2da = np.zeros((nym,nxm),dtype=np.float64)
h2dn = np.zeros((nym,nxm),dtype=np.float64)
K2dn = np.zeros((nym,nxm),dtype=np.float64)
Fx2dn = np.zeros((nym,nxm),dtype=np.float64)
Fy2dn = np.zeros((nym,nxm),dtype=np.float64)

f = 1.0
g = 1.0
H = 1.0

x = np.zeros(nxm)
y = np.zeros(nym)
hx = np.zeros(nxm*nym,dtype=np.float64)
ux = np.zeros(2*nxm*nym,dtype=np.float64)
wx = np.zeros(nxm*nym,dtype=np.float64)

sw = SWEqn(topo,quad,topo_q,lx,ly,f,g,8.0e-7)

startStep = int(sys.argv[1])
timeStride = 1
print 'starting from step: %d'%startStep
if startStep == 0:
	print 'initializing primitive variables...'
	for ey in np.arange(ny):
		for ex in np.arange(nx):
			inds0 = topo_q.localToGlobal0(ex,ey)
			inds1x = topo.localToGlobal1x(ex,ey)
			inds1y = topo.localToGlobal1y(ex,ey)
			for jj in np.arange(mp1*mp1):
				if jj%mp1 == m or jj/mp1 == m:
					continue
				i = inds0[jj]%nxm
				j = inds0[jj]/nxm
				x[i] = ex*dx + 0.5*dx*(quad.x[jj%mp1] + 1.0)
				y[j] = ey*dy + 0.5*dy*(quad.x[jj/mp1] + 1.0)
				ux[inds0[jj]] = velx_func(x[i],y[j])
				ux[inds0[jj]+shiftQuad] = vely_func(x[i],y[j])
				hx[inds0[jj]] = pres_func(x[i],y[j])

	Mxto1 = Xto1(topo,quad).M
	ui = Mxto1*ux
	Mxto2 = Xto2(topo,quad).M
	hi = Mxto2*hx

else:
	ufile = 'uf_%.4d'%startStep+'.npy'
	hfile = 'hf_%.4d'%startStep+'.npy'
	print 'loading primitive variables from files: ' + ufile + ' and: ' + hfile
	ui = np.load(ufile)
	hi = np.load(hfile)

	for ey in np.arange(ny):
		for ex in np.arange(nx):
			inds0 = topo_q.localToGlobal0(ex,ey)
			for jj in np.arange(mp1*mp1):
				if jj%mp1 == m or jj/mp1 == m:
					continue
				i = inds0[jj]%nxm
				j = inds0[jj]/nxm
				x[i] = ex*dx + 0.5*dx*(quad.x[jj%mp1] + 1.0)
				y[j] = ey*dy + 0.5*dy*(quad.x[jj/mp1] + 1.0)

qi = sw.diagnose_q(hi,ui)
Fi = sw.diagnose_F(hi,ui)
Ki = sw.diagnose_K(ui)

if startStep == 0:
	print 'plotting initial conditions...'
	for ey in np.arange(ny):
		for ex in np.arange(nx):
			inds0 = topo_q.localToGlobal0(ex,ey)
			inds0n = topo.localToGlobal0(ex,ey)
			inds1x = topo.localToGlobal1x(ex,ey)
			inds1y = topo.localToGlobal1y(ex,ey)
			inds2 = topo.localToGlobal2(ex,ey)
			for jj in np.arange(mp1*mp1):
				if jj%mp1 == m or jj/mp1 == m:
					continue
				i = inds0[jj]%nxm
				j = inds0[jj]/nxm
				ii = i%m
				jj = j%m

				u2dn[j][i] = 0.0
				v2dn[j][i] = 0.0
				Fx2dn[j][i] = 0.0
				Fy2dn[j][i] = 0.0
				for kk in np.arange(np1*n):
					u2dn[j][i] = u2dn[j][i] + ui[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
					v2dn[j][i] = v2dn[j][i] + ui[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]
					Fx2dn[j][i] = Fx2dn[j][i] + Fi[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
					Fy2dn[j][i] = Fy2dn[j][i] + Fi[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

				h2dn[j][i] = 0.0
				K2dn[j][i] = 0.0
				for kk in np.arange(n*n):
					h2dn[j][i] = h2dn[j][i] + hi[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]
					K2dn[j][i] = K2dn[j][i] + Ki[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]

				q2dn[j][i] = 0.0
				for kk in np.arange(np1*np1):
					q2dn[j][i] = q2dn[j][i] + qi[inds0n[kk]]*Njxi[ii,kk%np1]*Njxi[jj,kk/np1]

	Plot(x,y,h2dn,'h (initial)','pres_0000.png',0.0,0.0)
	Plot(x,y,u2dn,'u (initial)','velx_0000.png',0.0,0.0)
	Plot(x,y,v2dn,'v (initial)','vely_0000.png',0.0,0.0)
	Plot(x,y,q2dn,'w (initial)','vort_0000.png',0.0,0.0)
	Plot(x,y,K2dn,'K (initial)','kine_0000.png',0.0,0.0)
	Plot(x,y,Fx2dn,'Fx (initial)','Fx_0000.png',0.0,0.0)
	Plot(x,y,Fy2dn,'Fy (initial)','Fy_0000.png',0.0,0.0)
else:
	print 'plotting input fields...'
	for ey in np.arange(ny):
		for ex in np.arange(nx):
			inds0 = topo_q.localToGlobal0(ex,ey)
			inds1x = topo.localToGlobal1x(ex,ey)
			inds1y = topo.localToGlobal1y(ex,ey)
			inds2 = topo.localToGlobal2(ex,ey)
			for jj in np.arange(mp1*mp1):
				if jj%mp1 == m or jj/mp1 == m:
					continue
				i = inds0[jj]%nxm
				j = inds0[jj]/nxm
				ii = i%m
				jj = j%m

				u2dn[j][i] = 0.0
				v2dn[j][i] = 0.0
				for kk in np.arange(np1*n):
					u2dn[j][i] = u2dn[j][i] + ui[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
					v2dn[j][i] = v2dn[j][i] + ui[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

				h2dn[j][i] = 0.0
				for kk in np.arange(n*n):
					h2dn[j][i] = h2dn[j][i] + hi[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]


	Plot(x,y,h2dn,'h (restart)','pres_restart.png',0.0,0.0)
	Plot(x,y,u2dn,'u (restart)','velx_restart.png',0.0,0.0)
	Plot(x,y,v2dn,'v (restart)','vely_restart.png',0.0,0.0)

print 'initializing conservation terms...'
nsteps = 8000
#dt = 0.10*dx/2.0/n
dt = 0.10*dx/np.sqrt(g*H)/n
dt = dt/timeStride
nsteps = timeStride*nsteps

volC = np.zeros((nsteps+startStep*timeStride),dtype=np.float64)
pvC = np.zeros((nsteps+startStep*timeStride),dtype=np.float64)
peC = np.zeros((nsteps+startStep*timeStride),dtype=np.float64)
teC = np.zeros((nsteps+startStep*timeStride),dtype=np.float64)

M0 = Pmat(topo,quad).M
M1 = Umat(topo,quad).M
M0inv = la.inv(M0)
D10 = BoundaryMat10(topo).M
D01 = D10.transpose()
Curl = (2.0*nx/lx)*M0inv*D01*M1
wi = Curl*ui
vol0, pv0, pe0, te0 = TestConservation(topo,quad,lx,ly,f,g,hi,ui,qi,Ki,wi)
print 'initial volume              %10.8e'%vol0
print 'initial vorticity           %10.8e'%pv0
print 'initial potential enstrophy %10.8e'%pe0
print 'initial total energy        %10.8e'%te0

for step in np.arange(nsteps) + 1 + startStep*timeStride:
	print '\tsolving shallow water for time step %.4d'%step

	hf,uf = sw.solveRK2(hi,ui,dt)
	qf = sw.diagnose_q(hf,uf)
	Kf = sw.diagnose_K(uf)
	wf = Curl*uf

	if np.mod(step,10*timeStride) == 0:
		print '\tplotting height and velocity fields %.4d'%step
		for ey in np.arange(ny):
			for ex in np.arange(nx):
				inds0q = topo_q.localToGlobal0(ex,ey)
				inds0 = topo.localToGlobal0(ex,ey)
				#inds1x = topo.localToGlobal1x(ex,ey)
				#inds1y = topo.localToGlobal1y(ex,ey)
				inds2 = topo.localToGlobal2(ex,ey)
				for jj in np.arange(mp1*mp1):
					if jj%mp1 == m or jj/mp1 == m:
						continue
					i = inds0q[jj]%nxm
					j = inds0q[jj]/nxm
					ii = i%m
					jj = j%m

					#q2dn[j][i] = 0.0
					q2da[j][i] = 0.0
					for kk in np.arange(np1*np1):
						#q2dn[j][i] = q2dn[j][i] + qf[inds0[kk]]*Njxi[ii,kk%np1]*Njxi[jj,kk/np1]
						q2da[j][i] = q2da[j][i] + wf[inds0[kk]]*Njxi[ii,kk%np1]*Njxi[jj,kk/np1]

					#u2dn[j][i] = 0.0
					#v2dn[j][i] = 0.0
					#for kk in np.arange(np1*n):
					#	u2dn[j][i] = u2dn[j][i] + uf[inds1x[kk]]*Njxi[ii,kk%np1]*Ejxi[jj,kk/np1]
					#	v2dn[j][i] = v2dn[j][i] + uf[inds1y[kk]]*Ejxi[ii,kk%n]*Njxi[jj,kk/n]

					h2dn[j][i] = 0.0
					for kk in np.arange(n*n):
						h2dn[j][i] = h2dn[j][i] + hf[inds2[kk]]*Ejxi[ii,kk%n]*Ejxi[jj,kk/n]

		Plot(x,y,q2da,'w (numeric)','vort_%.4d'%(step/timeStride) + '.png',0.0,0.0)
		#Plot(x,y,q2dn,'q (numeric)','potv_%.4d'%(step/timeStride) + '.png',0.0,0.0)
		#Plot(x,y,u2dn,'u (numeric)','velx_%.4d'%(step/timeStride) + '.png',0.0,0.0)
		#Plot(x,y,v2dn,'v (numeric)','vely_%.4d'%(step/timeStride) + '.png',0.0,0.0)
		Plot(x,y,h2dn,'h (numeric)','pres_%.4d'%(step/timeStride) + '.png',0.0,0.0)

	print '\tdiagnosing conservation for time step %.4d'%step

	vol, pv, pe, te = TestConservation(topo,quad,lx,ly,f,g,hf,uf,qf,Kf,wf)

	print '\tvolume conservation:               %10.8e'%((vol - vol0)/vol0)
	print '\tvorticity conservation:            %10.8e'%(pv - pv0)
	print '\tpotential enstrophy conservation:  %10.8e'%((pe - pe0)/pe0)
	print '\ttotal energy conservation:         %10.8e'%((te - te0)/te0)

	volC[step-1] = (vol-vol0)/vol0
	pvC[step-1] = (pv-pv0)
	peC[step-1] = (pe-pe0)/pe0
	teC[step-1] = (te-te0)/te0

	np.save('volume_%d'%timeStride,volC[:step])
	np.save('vorticity_%d'%timeStride,pvC[:step])
	np.save('potentialEnstrophy_%d'%timeStride,peC[:step])
	np.save('energy_%d'%timeStride,teC[:step])

	hi[:] = hf[:]
	ui[:] = uf[:]

	if np.mod(step/timeStride,100) == 0:
		np.save('uf_%.4d'%(step/timeStride),uf)
		np.save('hf_%.4d'%(step/timeStride),hf)
