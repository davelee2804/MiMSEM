#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

cp = 1004.5
cv = 717.5

#istep=187
istep=30
en = 1.0*np.loadtxt('energetics.dat')[istep:,:]
nt = en.shape[0]

time = 120.0*(np.arange(en.shape[0]) + 1)
time = (1.0/60.0/60.0/24.0)*time

e0=en[0,0]+en[0,1]+en[0,2]+en[0,3]

plt.plot(time[:],(en[:,0]+en[:,1]+en[:,2]+en[:,3]-e0)/e0)
plt.title('total energy change vs. time')
plt.xlabel('time (days)')
#plt.savefig('energetics_total_energy_difference.png')
plt.show()

plt.semilogy(time[:],en[:,0]+en[:,1]+en[:,2]+en[:,3])
plt.semilogy(time[:],en[:,0])
plt.semilogy(time[:],en[:,1])
plt.semilogy(time[:],en[:,2])
plt.semilogy(time[:],en[:,3])
plt.title('energy vs. time')
plt.legend(['Total','K (horiz.)','K (vert.)','P','I'],loc='center left')
plt.xlabel('time (days)')
plt.savefig('energetics_energy_partition_2.png')
plt.show()

plt.plot(time[:],en[:,0]-en[0,0])
plt.plot(time[:],en[:,1]-en[0,1])
plt.plot(time[:],en[:,2]-en[0,2])
plt.plot(time[:],en[:,3]-en[0,3])
plt.plot(time[:],en[:,0]+en[:,1]+en[:,2]+en[:,3]-e0)
plt.legend(['K (horiz.)','K (vert.)','P','I','Total'],loc='lower left')
plt.ylabel('E-E(t=0)')
plt.xlabel('time (days)')
plt.savefig('energetics_energy_difference.png')
plt.show()

plt.semilogy(time[:],np.abs(en[:,0]+en[:,1]+en[:,2]+en[:,3]-e0)/e0)
plt.semilogy(time[:],np.abs(en[:,0]-en[0,0])/en[0,0])
plt.semilogy(time[:],np.abs(en[:,1]-en[0,1])/en[0,1])
plt.semilogy(time[:],np.abs(en[:,2]-en[0,2])/en[0,2])
plt.semilogy(time[:],np.abs(en[:,3]-en[0,3])/en[0,3])
#plt.title('energy vs. time')
plt.legend(['Total','K (horiz.)','K (vert.)','P','I'],loc='lower right')
plt.ylabel('|E-E(t=0)|/E(t=0)')
plt.xlabel('time (days)')
plt.savefig('energetics_energy_partition.png')
plt.show()

plt.plot(time[:],en[:,4])
plt.plot(time[:],en[:,5])
plt.plot(time[:],en[:,7])
#plt.title('energy exchanges vs. time')
plt.legend(['P to K','I to K (horiz)','I to K (vert)'],loc='lower left')
plt.xlabel('time (days)')
plt.ylabel('power ($kg\cdot m^2s^{-3}$)')
plt.ylim([-2.0e+15,+2.0e+15])
plt.savefig('energetics_power_exchanges.png')
plt.show()
