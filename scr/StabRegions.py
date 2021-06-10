#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'Reds'

nx = 301
ny = 301

re = np.linspace(-4.0,+4.0,nx)
im = np.linspace(-4.0,+4.0,nx)

IM,RE = np.meshgrid(re,im)

Lstab = np.zeros((nx,ny))
Nstab = np.zeros((nx,ny))

def stab_func(alpha,dt,re,im):
	c = (re+1.0j*im)*dt
	top = 1.0 + (1.0 - 2.0*alpha)*c + (0.5 - 2.0*alpha + alpha*alpha)*c*c
	bot = (1.0 - alpha*c)*(1.0 - alpha*c)
	A = top/bot
	Amag = np.sqrt(A.real*A.real + A.imag*A.imag)
	#return Amag
	if Amag < 1.0000001:
		return 1
	else:
		return 0

alpha_l = 0.5 + 0.5*np.sqrt(2.0)
alpha_n = 0.25

for jr in np.arange(nx):
	for ji in np.arange(nx):
		Lstab[jr,ji] = stab_func(alpha_l,1.0,re[jr],im[ji])
		Nstab[jr,ji] = stab_func(alpha_n,1.0,re[jr],im[ji])

levs = np.array([1.000001])

plt.contourf(RE,IM,Lstab,100)
plt.colorbar()
#plt.contour(RE,IM,Lstab, levs, colors='k')
plt.show()

plt.contourf(RE,IM,Nstab,100)
plt.colorbar()
#plt.contour(RE,IM,Nstab, levs, colors='k')
plt.show()
