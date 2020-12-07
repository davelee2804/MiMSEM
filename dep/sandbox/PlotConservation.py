#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

A=np.loadtxt('src2/output/conservation.dat').transpose()
time = A[0,:]

plt.semilogy(time,np.abs(A[1,:]))
plt.ylabel(r'$|h - h(t=0)|/h(t=0)$')
plt.xlabel('time (days)')
plt.savefig('conservation_mass.png')
plt.show()

plt.semilogy(time,np.abs(A[2,:]))
plt.ylabel(r'$|\omega - \omega(t=0)|$')
plt.xlabel('time (days)')
plt.savefig('conservation_vorticity.png')
plt.show()

plt.semilogy(time,np.abs(A[3,:]))
plt.ylabel(r'$|E - E(t=0)|/E(t=0)$')
plt.xlabel('time (days)')
plt.savefig('conservation_energy.png')
plt.show()
