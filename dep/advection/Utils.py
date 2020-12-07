import numpy as np
import matplotlib.pyplot as plt

from Basis import *
from Topo import *

def psi1(x,y):
	xl = x - np.pi - np.pi/3.0
	yo = y - np.pi
	return np.exp(-2.5*(xl*xl + yo*yo))

def psi2(x,y):
	xr = x - np.pi + np.pi/3.0
	yo = y - np.pi
	return np.exp(-2.5*(xr*xr + yo*yo))

def strm_func(x,y):
	return psi1(x,y) + psi2(x,y)

def velx_test(x,y):
	yo = y - np.pi
	return +5.0*yo*psi1(x,y) + 5.0*yo*psi2(x,y)

def velx_func(x,y):
	yo = y - 5.0
	eps = 2.0#0.5
	fy = (1.0 - yo*yo)/eps + 1.0
	return (2.0*yo/eps)*0.1/(np.cosh(fy)*np.cosh(fy))

def vely_test(x,y):
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	return -5.0*xl*psi1(x,y) - 5.0*xr*psi2(x,y)

def vely_func(x,y):
	return 0.0

def pres_test(x,y):
	f = 1.0
	g = 1.0
	H = 1.0
	return (f/g)*strm_func(x,y) + H

def pres_test_2(x,y):
	return 0.25*(np.cos(x-np.pi) + 1.0)*(np.cos(y-np.pi) + 1.0)

def pres_func(x,y):
	xo = x - 5.0
	yo = y - 5.0
	r2 = xo*xo + yo*yo
	eps = 2.0#0.5
	fy = (1.0 - yo*yo)/eps + 1.0
	return 1.0 + 0.1*np.tanh(fy) + 0.005*np.exp(-r2)

def vort_func(x,y):
	f = 1.0
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	yo = y - np.pi
	w1 = 10.0*(2.5*(xl*xl + yo*yo) - 1.0)*psi1(x,y)
	w2 = 10.0*(2.5*(xr*xr + yo*yo) - 1.0)*psi2(x,y)
	#return (w1 + w2 + f)/pres_func(x,y)
	return (w1 + w2 + f)/pres_test(x,y)

def qFx_func(x,y):
	f = 1.0
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	yo = y - np.pi
	w1 = 10.0*(2.5*(xl*xl + yo*yo) - 1.0)*psi1(x,y)
	w2 = 10.0*(2.5*(xr*xr + yo*yo) - 1.0)*psi2(x,y)
	#return (w1 + w2 + f)*velx_func(x,y)
	return (w1 + w2 + f)*velx_test(x,y)
	
def qFy_func(x,y):
	f = 1.0
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	yo = y - np.pi
	w1 = 10.0*(2.5*(xl*xl + yo*yo) - 1.0)*psi1(x,y)
	w2 = 10.0*(2.5*(xr*xr + yo*yo) - 1.0)*psi2(x,y)
	#return (w1 + w2 + f)*vely_func(x,y)
	return (w1 + w2 + f)*vely_test(x,y)

def dpdx_func(x,y):
	yo = y - np.pi
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	return -5.0*xl*psi1(x,y) - 5.0*xr*psi2(x,y)

def dpdx_func_2(x,y):
	return -0.25*np.sin(x-np.pi)*(np.cos(y-np.pi) + 1.0)

def d2pdxx_func(x,y):
	yo = y - np.pi
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	return -5.0*psi1(x,y) - 5.0*psi2(x,y) +25.0*xl*xl*psi1(x,y) + 25.0*xr*xr*psi2(x,y)

def dpdy_func(x,y):
	yo = y - np.pi
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	return -5.0*yo*psi1(x,y) - 5.0*yo*psi2(x,y)

def dpdy_func_2(x,y):
	return -0.25*(np.cos(x-np.pi) + 1.0)*np.sin(y-np.pi)

def d2pdyy_func(x,y):
	yo = y - np.pi
	xl = x - np.pi - np.pi/3.0
	xr = x - np.pi + np.pi/3.0
	return -5.0*psi1(x,y) - 5.0*psi2(x,y) + 25.0*yo*yo*psi1(x,y) + 25.0*yo*yo*psi2(x,y)

def Plot(x,y,dat,title,fname,zmin,zmax):
	if np.abs(zmin) > 0.01 or np.abs(zmax) > 0.01:
		levs = np.linspace(zmin,zmax,101,endpoint=True)
		plt.contourf(x,y,dat,levs)
	else:
		plt.contourf(x,y,dat,101)
	plt.colorbar()
	plt.title(title)
	plt.savefig(fname)
	plt.clf()

def TestConservation(topo,quad,lx,ly,f,g,h,u,q,k,w):
	det = 0.5*lx/topo.nx*0.5*ly/topo.ny
	shift1Form = topo.nx*topo.ny*topo.n*topo.n

	n = topo.n
	np1 = topo.n + 1
	mp1 = quad.n + 1

	edge = LagrangeEdge(n)
	Ejxi = np.zeros((mp1,n),dtype=np.float64)
	for j in np.arange(n):
		for i in np.arange(mp1):
			Ejxi[i,j] = edge.eval(quad.x[i],j)

	node = LagrangeNode(n)
	Njxi = np.zeros((mp1,np1),dtype=np.float64)
	for j in np.arange(np1):
		for i in np.arange(mp1):
			Njxi[i,j] = node.eval(quad.x[i],j)

	volume = 0.0
	potVort = 0.0
	potEnst = 0.0
	totEner = 0.0

	for ex in np.arange(topo.nx):
		for ey in np.arange(topo.ny):
			inds0 = topo.localToGlobal0(ex,ey)
			inds2 = topo.localToGlobal2(ex,ey)
			inds1x = topo.localToGlobal1x(ex,ey)
			inds1y = topo.localToGlobal1x(ex,ey) + shift1Form

			volume_e = 0.0
			potVort_e = 0.0
			potEnst_e = 0.0
			totEner_e = 0.0
			for jj in np.arange(mp1*mp1):
				jx = jj%mp1
				jy = jj/mp1
				wx = quad.w[jx]
				wy = quad.w[jy]
				wt = wx*wy

				qq = 0.0
				wq = 0.0
				for ii in np.arange(np1*np1):
					ix = ii%np1
					iy = ii/np1
					qq = qq + q[inds0[ii]]*Njxi[jx,ix]*Njxi[jy,iy]
					wq = wq + w[inds0[ii]]*Njxi[jx,ix]*Njxi[jy,iy]

				hq = 0.0
				kq = 0.0
				for ii in np.arange(n*n):
					ix = ii%n
					iy = ii/n
					hq = hq + h[inds2[ii]]*Ejxi[jx,ix]*Ejxi[jy,iy]
					kq = kq + k[inds2[ii]]*Ejxi[jx,ix]*Ejxi[jy,iy]

				volume_e = volume_e + wt*hq
				potVort_e = potVort_e + wt*wq
				potEnst_e = potEnst_e + 0.5*wt*qq*qq*hq
				totEner_e = totEner_e + wt*(hq*kq + 0.5*g*hq*hq)

			volume = volume + volume_e
			potVort = potVort + potVort_e
			potEnst = potEnst + potEnst_e
			totEner = totEner + totEner_e

	return det*volume, det*potVort, det*potEnst, det*totEner

def TestDiagnosticError(topo, quad, dx, dy, qi, Fi, Ki):
	n = topo.n
	np1 = topo.n + 1
	mp1 = quad.n + 1

	edge = LagrangeEdge(n)
	Ejxi = np.zeros((mp1,n),dtype=np.float64)
	for j in np.arange(n):
		for i in np.arange(mp1):
			Ejxi[i,j] = edge.eval(quad.x[i],j)

	node = LagrangeNode(n)
	Njxi = np.zeros((mp1,np1),dtype=np.float64)
	for j in np.arange(np1):
		for i in np.arange(mp1):
			Njxi[i,j] = node.eval(quad.x[i],j)

	err_q = 0.0
	err_Fx = 0.0
	err_Fy = 0.0
	err_K = 0.0
	norm_q = 0.0
	norm_Fx = 0.0
	norm_Fy = 0.0
	norm_K = 0.0

	for ex in np.arange(topo.nx):
		for ey in np.arange(topo.ny):
			inds0 = topo.localToGlobal0(ex,ey)
			inds1x = topo.localToGlobal1x(ex,ey)
			inds1y = topo.localToGlobal1y(ex,ey)
			inds2 = topo.localToGlobal2(ex,ey)

			for jj in np.arange(mp1*mp1):
				jx = jj%mp1
				jy = jj/mp1
				wx = quad.w[jx]
				wy = quad.w[jy]
				wt = wx*wy
				x = ex*dx + 0.5*dx*(quad.x[jj%mp1] + 1.0)
				y = ey*dy + 0.5*dy*(quad.x[jj/mp1] + 1.0)

				qq = 0.0
				for ii in np.arange(np1*np1):
					ix = ii%np1
					iy = ii/np1
					qq = qq + qi[inds0[ii]]*Njxi[jx,ix]*Njxi[jy,iy]

				Fxq = 0.0
				Fyq = 0.0
				for kk in np.arange(np1*n):
					Fxq = Fxq + Fi[inds1x[kk]]*Njxi[jx,kk%np1]*Ejxi[jy,kk/np1]
					Fyq = Fyq + Fi[inds1y[kk]]*Ejxi[jx,kk%n]*Njxi[jy,kk/n]

				Kq = 0.0
				for ii in np.arange(n*n):
					ix = ii%n
					iy = ii/n
					Kq = Kq + Ki[inds2[ii]]*Ejxi[jx,ix]*Ejxi[jy,iy]

				qa = vort_func(x,y)
				Fxa = pres_func(x,y)*velx_func(x,y)
				Fya = pres_func(x,y)*vely_func(x,y)
				Ka = 0.5*(velx_func(x,y)*velx_func(x,y) + vely_func(x,y)*vely_func(x,y))

				err_q = err_q + wt*(qq-qa)*(qq-qa)
				err_Fx = err_Fx + wt*(Fxq-Fxa)*(Fxq-Fxa)
				err_Fy = err_Fy + wt*(Fyq-Fya)*(Fyq-Fya)
				err_K = err_K + wt*(Kq-Ka)*(Kq-Ka)

				norm_q = norm_q + wt*qa*qa
				norm_Fx = norm_Fx + wt*Fxa*Fxa
				norm_Fy = norm_Fy + wt*Fya*Fya
				norm_K = norm_K + wt*Ka*Ka

	return err_q/norm_q, err_Fx/norm_Fx, err_Fy/norm_Fy, err_K/norm_K
