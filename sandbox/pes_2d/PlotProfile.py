#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

ETA_T = 0.2
GAMMA = 0.005
GRAVITY = 9.80616
RD = 287.04
T0 = 288.0
DELTA_T = 480000.0

Bk = np.array([0.05034551, 0.10365252, 0.16189536, 0.22606120, 0.29615005,\
               0.37413623, 0.45705824, 0.54392892, 0.63376111, 0.72260612,\
               0.80651530, 0.88153998, 0.94274432, 0.98519250, 1.00000000])

NK = 15

t_avg = np.zeros(NK)
for ii in np.arange(NK):
	eta = Bk[NK-ii-1]
	if eta < ETA_T:
		t_avg[ii] = T0*np.power(eta, RD*GAMMA/GRAVITY) + DELTA_T*np.power(ETA_T - eta, 5.0)
	else:
		t_avg[ii] = T0*pow(eta, RD*GAMMA/GRAVITY)

plt.plot(np.arange(NK),t_avg)
plt.show()
