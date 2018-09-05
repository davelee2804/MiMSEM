#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

en = 1.0*np.loadtxt('output/energetics.dat')

time = 60.0*(np.arange(en.shape[0]) + 1)

e0=en[0,0]+en[0,1]+en[0,2]+en[0,3]

plt.plot(time/60.0/60.0,(en[:,0]+en[:,1]+en[:,2]+en[:,3])/e0)
plt.title('total energy change vs. time')
plt.xlabel('time (hrs)')
plt.show()

plt.semilogy(time/60.0/60.0,en[:,0]+en[:,1]+en[:,2]+en[:,3])
plt.semilogy(time/60.0/60.0,en[:,0])
plt.semilogy(time/60.0/60.0,en[:,1])
plt.semilogy(time/60.0/60.0,en[:,2])
plt.semilogy(time/60.0/60.0,en[:,3])
plt.title('energy vs. time')
plt.legend(['total','K (horiz.)','K (vert.)','P','I'],loc='center left')
plt.xlabel('time (hrs)')
plt.show()

plt.plot(time/60.0/60.0,en[:,4])
plt.plot(time/60.0/60.0,en[:,5])
plt.plot(time/60.0/60.0,en[:,7])
plt.title('energy exchanges vs. time')
plt.legend(['K to P','K to I (horiz)','K to I (vert)'],loc='center left')
plt.xlabel('time (hrs)')
plt.show()
