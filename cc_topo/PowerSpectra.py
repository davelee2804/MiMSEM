#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
from pyshtools.expand import SHExpandDH
from pyshtools.spectralanalysis import spectrum

n = 0
#nLat = 3*4*12
nLat = 3*4*32
nLon = 2*nLat
lon = np.zeros((nLat*nLon),dtype=np.float64)
lat = np.zeros((nLat*nLon),dtype=np.float64)
ke = np.zeros((nLat*nLon),dtype=np.float64)

for ii in np.arange(24):
	filename = 'src2/output/gtol_%.4u.txt'%ii
	A = np.loadtxt(filename).transpose()
	na = A.shape[1]
	lon[n:n+na] = A[0,:]
	lat[n:n+na] = A[1,:]
	ke[n:n+na] = A[2,:]
	n = n + na

ke2 = np.zeros((nLat,nLon),dtype=np.float64)

dTheta = 2.0*np.pi/nLon
for ii in np.arange(n):
	ix = int((lon[ii] + 1.0*np.pi)/dTheta)
	iy = int((lat[ii] + 0.5*np.pi)/dTheta)
	if np.abs(ke2[iy,ix]) > 1.0e-6:
		print 'ERROR!'
	ke2[iy,ix] = ke[ii]

#inds = np.argsort(lat)
#lon = lon[inds]
#lat = lat[inds]
#ke = ke[inds]

#for ii in np.arange(nLat):
#	inds = np.argsort(lat[ii*nLon:(ii+1)*nLon])
#	lon[ii*nLon:(ii+1)*nLon] = lon[ii*nLon+inds]
#	lat[ii*nLon:(ii+1)*nLon] = lat[ii*nLon+inds]
#	ke[ii*nLon:(ii+1)*nLon] = ke[ii*nLon+inds]

print n
print nLon*nLat
print (1.0*n)/(1.0*nLon*nLat)
print (180/np.pi)*np.max(lat)
print (180/np.pi)*np.min(lat)

plt.plot((1.0/np.pi)*lon,(1.0/np.pi)*lat,'.')
plt.xlim([-1.0,+1.0])
plt.ylim([-0.5,+0.5])
plt.show()

levs = np.logspace(-8,5,200)

xx = np.linspace(-1.0*np.pi,+1.0*np.pi,nLon)
yy = np.linspace(-0.5*np.pi,+0.5*np.pi,nLat)
XX,YY=np.meshgrid(xx,yy)
#plt.contourf(XX, YY, ke2, 100, locator=ticker.LogLocator())
plt.contourf(XX, YY, ke2, norm=LogNorm(), levels=levs)
plt.colorbar(orientation='horizontal',ticks=np.array([1.0e-8,1.0e-6,1.0e-4,1.0e-2,0.0,1.0e2,1.0e4]))
plt.savefig('kinetic_energy_galewsky_7days.png')
plt.show()

coeffs = SHExpandDH(ke2, sampling=2)
power = spectrum(coeffs, unit='per_l')
nl = coeffs.shape[1]/2
plt.loglog(np.arange(nl), power[:nl])
plt.loglog(np.arange(nl-6)+6, 1.0e+7*np.power(np.arange(nl-6)+6,-3.0))
plt.loglog([50.0,50.0],[1.0e+5,1.0e-1])
plt.savefig('ke_spectra_galewsky_7days.png')
plt.show()
