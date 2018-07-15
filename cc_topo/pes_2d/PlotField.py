#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

rad_earth=6371220.0
a=np.sqrt(0.2)
qx=np.array([-1.0,-a,+a,+1.0])

n=3
ne=64
nz=int(sys.argv[2])
dx=2.0*np.pi*rad_earth/ne
x = np.zeros(n*ne)
for ii in np.arange(ne):
	for jj in np.arange(n):
		x[ii*n+jj] = ii*dx + 0.5*dx*(qx[jj]+1.0)

fname=sys.argv[1]
f=np.loadtxt(fname).reshape(nz,n*ne)

zr=100000.0*np.array([0.05034551, 0.10365252, 0.16189536, 0.22606120, 0.29615005,\
                0.37413623, 0.45705824, 0.54392892, 0.63376111, 0.72260612,\
                0.80651530, 0.88153998, 0.94274432, 0.98519250,1.0])
z=zr[::-1]

levs = np.linspace(226.0,408.0,101)

print x.shape
print z.shape
print f.shape
#plt.contourf(x,z[:nz],f,100)
plt.contourf(x,zr[:nz],f,100)
#plt.contourf(x,zr[:nz],f,levs)
plt.colorbar(orientation='horizontal')
plt.show()
